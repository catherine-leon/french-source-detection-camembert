#!/usr/bin/env python3
"""Pool-based active learning: CamemBERT / random / uncertainty / diversity.
Usage: python active_learning.py --data-dir . --output-dir al_outputs
"""
import argparse, gc, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


def load(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    return [' '.join(r['tokens']) for r in rows], np.array([int(any(tag != 'O' for tag in r['labels'])) for r in rows])


class WeightedTrainer(Trainer):
    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        output = model(**inputs)
        loss = CrossEntropyLoss(weight=self.weights.to(output.logits.device))(output.logits, labels)
        return (loss, output) if return_outputs else loss


def evaluate(logits_labels):
    logits, labels = logits_labels
    pred = np.argmax(logits, axis=1)
    return {'f1': f1_score(labels, pred, zero_division=0), 'accuracy': accuracy_score(labels, pred)}


def release(trainer):
    trainer.model.to('cpu')
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probabilities(trainer, dataset):
    logits = np.asarray(trainer.predict(dataset).predictions)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def make_dataset(texts, labels):
    dataset = Dataset.from_dict({'text': list(texts), 'label': np.asarray(labels).tolist()})
    return dataset.map(lambda b: tokenizer(b['text'], truncation=True, max_length=256),
                       batched=True, remove_columns=['text'])


def train(labeled, seed, directory):
    set_seed(seed)
    counts = np.bincount(y_train[labeled], minlength=2)
    if min(counts) == 0:
        raise ValueError(f'Both classes required, got {counts}')
    weights = torch.tensor(len(labeled) / (2 * counts), dtype=torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    training_args = TrainingArguments(
        output_dir=str(directory / 'checkpoints'), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5, eval_strategy='epoch', save_strategy='epoch',
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='f1', greater_is_better=True,
        seed=seed, data_seed=seed, report_to='none', disable_tqdm=True)
    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=make_dataset([x_train[i] for i in labeled], y_train[labeled]),
        eval_dataset=valid_dataset, data_collator=collator,
        compute_metrics=evaluate, weights=weights)
    trainer.train()
    p = probabilities(trainer, valid_dataset)
    threshold, f1 = max(((float(t), float(f1_score(y_valid, p > t, zero_division=0)))
                         for t in np.arange(.30, .91, .05)), key=lambda x: x[1])
    trainer.save_model(str(directory / 'model'))
    result = {'seed': seed, 'strategy': directory.parent.name, 'n_labeled': len(labeled),
              'valid_f1': f1, 'threshold': threshold,
              'valid_accuracy': float(accuracy_score(y_valid, p > threshold)),
              'class_0': int(counts[0]), 'class_1': int(counts[1])}
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2))
    release(trainer)
    # The best model is already in model/; Trainer checkpoints are redundant.
    shutil.rmtree(directory / 'checkpoints', ignore_errors=True)
    return result


def inference_trainer(directory):
    model = AutoModelForSequenceClassification.from_pretrained(directory / 'model')
    ta = TrainingArguments(output_dir=str(directory / 'inference'),
                           per_device_eval_batch_size=args.batch_size,
                           report_to='none', disable_tqdm=True)
    return Trainer(model=model, args=ta, data_collator=collator)


def select(strategy, directory, pool, seed, budget, n_samples):
    if strategy == 'random':
        return np.random.default_rng(seed + budget).choice(pool, n_samples, replace=False).tolist()
    trainer = inference_trainer(directory)
    dataset = make_dataset([x_train[i] for i in pool], np.zeros(len(pool), dtype=int))
    p = probabilities(trainer, dataset)
    uncertainty = np.abs(p - 0.5)
    if strategy == 'uncertainty':
        positions = np.argsort(uncertainty, kind='stable')[:n_samples]
    else:
        model = trainer.model
        model.eval()
        device = next(model.parameters()).device
        vectors = []
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
        with torch.inference_mode():
            for batch in loader:
                batch.pop('labels', None)
                batch = {key: value.to(device) for key, value in batch.items()}
                vectors.append(model.base_model(**batch).last_hidden_state[:, 0, :].cpu().numpy())
        vectors = np.concatenate(vectors, axis=0)
        clusters = KMeans(n_clusters=n_samples, random_state=seed + budget, n_init=10).fit_predict(vectors)
        positions = np.array([np.where(clusters == c)[0][np.argmin(uncertainty[clusters == c])]
                              for c in range(n_samples)])
    selected = pool[positions].tolist()
    release(trainer)
    return selected


def summarize(root):
    rows = [json.loads(p.read_text()) for p in root.glob('seed_*/**/metrics.json')]
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(['seed', 'strategy', 'n_labeled'])
    df.to_csv(root / 'results.csv', index=False)
    agg = df.groupby(['strategy', 'n_labeled'])['valid_f1'].agg(['mean', 'std', 'count']).reset_index()
    agg.to_csv(root / 'summary.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in agg.groupby('strategy'):
        group = group.sort_values('n_labeled')
        ax.plot(group.n_labeled, group['mean'], marker='o', label=name)
        if group['count'].min() >= 2:
            ax.fill_between(group.n_labeled.to_numpy(),
                            (group['mean'] - group['std']).to_numpy(),
                            (group['mean'] + group['std']).to_numpy(), alpha=.15)
    ax.set(xlabel='Number of labeled sentences', ylabel='Validation F1',
           title='Active Learning learning curves')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=200)
    plt.close(fig)


def main():
    global args, tokenizer, collator, x_train, y_train, y_valid, valid_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output-dir', default='al_outputs')
    parser.add_argument('--model', default='camembert-base')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    parser.add_argument('--max-labeled', type=int, default=750)
    parser.add_argument('--initial-size', type=int, default=50)
    parser.add_argument('--query-size', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--final-test', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load(Path(args.data_dir) / 'train.json')
    x_valid, y_valid = load(Path(args.data_dir) / 'valid.json')
    x_test, y_test = load(Path(args.data_dir) / 'test.json')
    if args.max_labeled > len(x_train):
        raise ValueError('Budget exceeds training set size')
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU (CPU)', flush=True)
    print('Dataset sizes:', len(x_train), len(x_valid), len(x_test), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_dataset = make_dataset(x_valid, y_valid)
    (root / 'config.json').write_text(json.dumps(vars(args), indent=2))
    for seed in args.seeds:
        ids = np.arange(len(x_train))
        initial, _ = train_test_split(ids, train_size=args.initial_size,
                                      stratify=y_train, random_state=seed)
        for strategy in ('random', 'uncertainty', 'diversity'):
            base = root / f'seed_{seed}' / strategy
            state_path = base / 'state.json'
            if state_path.exists():
                labeled = np.array(json.loads(state_path.read_text())['labeled'], dtype=int)
            else:
                labeled = np.array(initial, dtype=int)
            while True:
                n = len(labeled)
                directory = base / f'n_{n}'
                directory.mkdir(parents=True, exist_ok=True)
                # Current-round model is needed for selection or final test.
                # Earlier rounds have already been recorded in state.json.
                if not (directory / 'metrics.json').exists() or not (directory / 'model' / 'config.json').exists():
                    print(f'TRAIN seed={seed}, strategy={strategy}, n={n}', flush=True)
                    outcome = train(labeled, seed, directory)
                    print('Validation F1:', round(outcome['valid_f1'], 4), flush=True)
                    summarize(root)
                else:
                    print(f'RESUME seed={seed}, strategy={strategy}, n={n}', flush=True)
                if n >= args.max_labeled:
                    if args.final_test and not (directory / 'test_metrics.json').exists():
                        trainer = inference_trainer(directory)
                        p = probabilities(trainer, make_dataset(x_test, y_test))
                        threshold = json.loads((directory / 'metrics.json').read_text())['threshold']
                        pred = p > threshold
                        (directory / 'test_metrics.json').write_text(json.dumps({
                            'seed': seed, 'strategy': strategy, 'n_labeled': n,
                            'test_f1': float(f1_score(y_test, pred, zero_division=0)),
                            'test_accuracy': float(accuracy_score(y_test, pred)),
                            'threshold_from_validation': threshold}, indent=2))
                        release(trainer)
                    break
                pool = np.setdiff1d(ids, labeled)
                chosen = select(strategy, directory, pool, seed, n,
                                min(args.query_size, args.max_labeled - n))
                assert len(chosen) == len(set(chosen)) and np.isin(chosen, pool).all()
                labeled = np.concatenate([labeled, np.array(chosen, dtype=int)])
                # Persist selection before deleting the model. Atomic replacement
                # avoids a partially written state file if execution stops.
                tmp_state = state_path.with_suffix('.json.tmp')
                tmp_state.write_text(json.dumps({'labeled': labeled.tolist(),
                                                 'last_selected': chosen}, indent=2))
                tmp_state.replace(state_path)
                # This model is no longer needed: next round retrains from
                # pretrained CamemBERT, and metrics are already saved.
                shutil.rmtree(directory / 'model', ignore_errors=True)
                shutil.rmtree(directory / 'checkpoints', ignore_errors=True)
                print('Selected:', len(chosen), '| Total labeled:', len(labeled), flush=True)
    summarize(root)
    print('DONE:', root.resolve(), flush=True)


if __name__ == '__main__':
    main()
