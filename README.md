# French Source-Reference Detection with CamemBERT

Sentence-level source-reference detection in French parliamentary texts, combining supervised fine-tuning of **CamemBERT** with an evaluation of **active learning** strategies for reducing annotation requirements.

## Task and data

The task is **binary sentence classification**: predict whether a sentence contains at least one reference to a source. The original data include token-level annotations; these are converted to a sentence label (`1` if at least one token is tagged as a source reference, `0` otherwise). This project does **not** perform token-level entity extraction.

| Split | Sentences | Positive |
|---|---:|---:|
| Train | 1,500 | 958 |
| Validation | 200 | 128 |
| Test | 90 | 57 |

Data are provided as JSONL files (`data/train.json`, `data/valid.json`, `data/test.json`).

## Part 1 — Supervised CamemBERT fine-tuning

[`source_reference_classification.py`](source_reference_classification.py) fine-tunes `camembert-base` for binary classification. The training procedure uses class-weighted cross-entropy, selects the best checkpoint based on validation F1, and chooses a classification threshold using the validation split. Final performance is evaluated on the held-out test split.

**Test F1: approximately 0.974**, trained on the full 1,500-sentence training set.

## Part 2 — Active learning

[`active_learning.py`](active_learning.py) compares three ways to select sentences for annotation:

- **Random sampling:** a random acquisition baseline.
- **Uncertainty sampling:** selects examples where the current classifier is least confident.
- **Diversity-based sampling:** combines representation-space clustering (KMeans on CamemBERT embeddings) with uncertainty-based selection.

Each experiment starts with **50 labeled sentences** and acquires **50 additional sentences per round**, up to **750 labeled sentences**. Results are repeated with three random seeds (**42, 123, 2026**). Models are retrained and evaluated on the validation set throughout the learning process; final test evaluation is performed at the end.

### Results

Mean held-out **test F1** at 750 labeled sentences (three seeds):

| Sampling strategy | Mean test F1 |
|---|---:|
| Random | 0.9641 |
| Uncertainty | 0.9667 |
| Diversity + uncertainty | **0.9760** |

The diversity-based strategy achieved the highest mean test F1 in these runs, using **half of the available training pool**. Its mean score was comparable to the full-data supervised model, although this is **not evidence of statistical equivalence or a significant improvement**: the test split contains only 90 sentences, and the comparison uses three random seeds.

![Active learning curves](results/learning_curves.png)

The learning curves summarize **validation F1** across annotation budgets; the table above reports **held-out test F1** at the final budget. Raw results and experiment settings are available in [`results/`](results/).

## Repository structure

```text
.
├── data/
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── results/
│   ├── results.csv
│   ├── summary.csv
│   ├── learning_curves.png
│   └── config.json
├── source_reference_classification.py
├── active_learning.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ and a GPU are recommended for model training.

```bash
git clone https://github.com/catherine-leon/french-source-detection-camembert.git
cd french-source-detection-camembert
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment using `.venv\Scripts\activate` instead.

## Reproduction

Run supervised fine-tuning:

```bash
python source_reference_classification.py --data-dir data --output-dir baseline_outputs
```

Run the active learning experiments:

```bash
python active_learning.py --data-dir data --output-dir al_outputs --final-test
```

The active learning script saves iteration-level metrics, aggregate learning curves, and final test metrics. Reproducing the complete experiment involves many model-training runs and can take substantial GPU time.

## Acknowledgements

The original course project and its dataset were developed within the framework of **DAPEPANE**. The work presented here extends that initial project with a systematic comparison of active learning strategies.
