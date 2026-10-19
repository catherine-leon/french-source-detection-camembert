#!/usr/bin/env python3
"""CamemBERT baseline for binary French source-reference detection.

Converted from classification_binaire.ipynb; training and evaluation
methodology preserved. Input files are JSON Lines (one record per line).

Example:
    python baseline.py --data-dir . --output-dir results_classif
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from datasets import Dataset
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# HYPERPARAMÈTRES
CONFIG = {
    'model':         'camembert-base',
    'learning_rate': 2e-5,
    'batch_size':    16,
    'max_length':    256,
    'epochs':        5,      # maximum : la meilleure époque est choisie sur la validation
    'seed':          42,
    'class_weights': 'balanced',  # 'balanced' (calculés sur le train) ou None
    'thresholds':    np.arange(0.30, 0.91, 0.05),  # seuils testés sur la validation
}


def load_jsonl(path):
    """Charge un fichier JSONL (un objet JSON par ligne)."""
    data = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Avertissement] {path}, ligne {i+1} ignorée : {e}")
    return data


def prepare_data(dataset):
    """Label binaire = 1 si au moins un token porte une étiquette IOB différente de 'O'."""
    X, y = [], []
    for item in dataset:
        X.append(' '.join(item['tokens']))
        y.append(1 if any(l != 'O' for l in item['labels']) else 0)
    return X, y


def count_classes(y, name='dataset'):
    """Affiche la distribution des classes 0 / 1."""
    counter, total = Counter(y), len(y)
    print(f'\n{name.upper()}')
    print('=' * 30)
    for cls in sorted(counter):
        print(f'  classe {cls} : {counter[cls]} ({counter[cls] / total * 100:.2f}%)')
    print(f'  total : {total}')


def compute_class_weights(y):
    """Poids 'balanced' : n_total / (n_classes * n_classe). La classe minoritaire pèse plus."""
    counts = np.bincount(y, minlength=2)
    return torch.tensor(len(y) / (2 * counts), dtype=torch.float)


def compute_metrics(eval_pred):
    """Métriques sur la validation à chaque époque (seuil 0.5, sert à choisir la meilleure époque)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds),
    }


class WeightedTrainer(Trainer):
    """Trainer HuggingFace avec Cross-Entropy Loss pondérée par classe."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss    = CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_probs(trainer, dataset):
    """Probabilités de la classe 1 et vrais labels."""
    output = trainer.predict(dataset)
    probs  = F.softmax(torch.tensor(output.predictions), dim=1)[:, 1].numpy()
    return probs, output.label_ids


def choose_threshold(probs, y_true, thresholds):
    """Seuil qui maximise le F1 (classe 1) sur la validation."""
    scores = [(t, f1_score(y_true, (probs > t).astype(int))) for t in thresholds]
    for t, s in scores:
        print(f'  seuil {t:.2f} : F1 = {s:.4f}')
    return max(scores, key=lambda x: x[1])
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='Directory with train.json, valid.json, test.json')
    parser.add_argument('--output-dir', type=Path, default=Path('results_classif'), help='Training checkpoints and confusion matrix')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(CONFIG['seed'])

    train_data = load_jsonl(args.data_dir / 'train.json')
    dev_data   = load_jsonl(args.data_dir / 'valid.json')
    test_data  = load_jsonl(args.data_dir / 'test.json')

    X_train, y_train = prepare_data(train_data)
    X_dev,   y_dev   = prepare_data(dev_data)
    X_test,  y_test  = prepare_data(test_data)

    count_classes(y_train, 'train')
    count_classes(y_dev,   'validation')
    count_classes(y_test,  'test')

    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
    dev_dataset   = Dataset.from_dict({'text': X_dev,   'label': y_dev})
    test_dataset  = Dataset.from_dict({'text': X_test,  'label': y_test})
    tokenizer = CamembertTokenizer.from_pretrained(CONFIG['model'])

    def tokenize(example):
        return tokenizer(example['text'], truncation=True, max_length=CONFIG['max_length'])

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
    dev_dataset   = dev_dataset.map(tokenize, batched=True, remove_columns=['text'])
    test_dataset  = test_dataset.map(tokenize, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = CamembertForSequenceClassification.from_pretrained(CONFIG['model'], num_labels=2)

    class_weights = compute_class_weights(y_train) if CONFIG['class_weights'] == 'balanced' else None
    print('Poids des classes :', class_weights)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=50,
        seed=CONFIG['seed'],
        report_to='none',
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    dev_probs, dev_true = get_probs(trainer, dev_dataset)
    threshold, dev_f1 = choose_threshold(dev_probs, dev_true, CONFIG['thresholds'])
    print(f'\nSeuil retenu : {threshold:.2f} (F1 validation = {dev_f1:.4f})')
    test_probs, y_true = get_probs(trainer, test_dataset)
    y_pred = (test_probs > threshold).astype(int)

    print(f'TEST — seuil {threshold:.2f}')
    print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')
    print(f'F1       : {f1_score(y_true, y_pred):.4f}')
    print('\nRapport de classification :\n')
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion (test)')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'confusion_matrix.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
