# Contributing results to GENEB

Repository: https://github.com/DeepPavlov/geneb  
Task data (pinned): https://huggingface.co/datasets/DeepPavlov/geneb-tasks  
Leaderboard: https://huggingface.co/spaces/DeepPavlov/geneb-leaderboard

To add a model, implement an embedding extractor, run the reference harness on the pinned
task data, and open a pull request with the resulting submission file, extractor module, and
a short model card. Evaluation runs on your hardware; weights and embeddings remain local.

Community entries are marked `self-reported` in submission metadata after review.

---

## Quick start

```bash
# 1. Implement harness/extractors/<module>.py (subclass BaseEmbeddingExtractor)
# 2. Download pinned task CSVs
python3 -m pip install "huggingface_hub>=0.24"
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data

# 3. Run the harness (example)
python3 harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --url https://huggingface.co/org/my-model \
  --data_dir ./GENEB_data --device cuda --submitted_by "Your Name"

# 4. Validate and open a PR with:
#    submissions/<model_id>.json, harness/extractors/<module>.py, model_cards/<model_id>.md
python3 tools/validate_submission.py submissions/my-model-300m.json
```

Use `--limit N` to evaluate the first *N* tasks before a full benchmark run.

---

## 1. Extractor

Add one module under `harness/extractors/` that implements `extract_embeddings` and returns
a matrix of shape `(n_sequences, embedding_dim)`:

```python
from .base import BaseEmbeddingExtractor
import numpy as np

class MyModelExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device: str = "cpu"):
        self.name_model = name_model
        self.device = device
        # load model and tokenizer

    def extract_embeddings(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        ...
```

The repository includes reference extractors only where needed for testing; production
models are added through contributions. Publishing the extractor alongside your
submission enables independent reproduction of your reported metrics.

---

## 2. Dataset

GENEB uses **100** CSV files named `<task_id>.csv` (see `benchmark/benchmark_spec.json`),
with columns `text`, `label`, and `split` (`train` or `test`).

Download the revision pinned in the spec:

```bash
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data
```

Use the released train/test partitions without relabeling or resplitting. On the Hugging
Face dataset page, each subset corresponds to one task; train and test are separate splits.

---

## 3. Run the harness

```bash
python3 harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --data_dir ./GENEB_data --device cuda
```

For each task, the harness (i) embeds train and test sequences, (ii) fits a logistic
regression probe with the settings in `benchmark_spec.json` (seeds
`[13, 17, 42, 123, 997]`, `max_iter=1000`, `n_jobs` as specified), and (iii) reports
mean MCC, accuracy, and macro-F1 over seeds for regimes `full`, `1shot`, and `10shot`.
Output is written to `submissions/<model_id>.json`.

**Legacy workflow:** if you already have per-task `results_*.json` from an earlier pipeline
with the same protocol, convert them with:

```bash
python3 tools/pack_submission.py --raw_dir <results_dir> --model_id <id> ...
```

---

## 4. Model card

Add `model_cards/<model_id>.md` following [model_cards/README.md](model_cards/README.md).
Include architecture, training data, zero-shot status, and any known overlap between
pretraining data and benchmark tasks.

---

## 5. Pull request

Required files:

- `submissions/<model_id>.json`
- `harness/extractors/<module>.py`
- `model_cards/<model_id>.md`

CI runs `validate_submission.py` and rebuilds leaderboard JSON in dry-run. After merge to
`main`, the public Space is updated automatically (requires repository secrets).

---

## What CI verifies

- Valid JSON and filename ↔ `model_id` consistency  
- All 100 tasks present in each regime (`full`, `1shot`, `10shot`)  
- Metrics within valid ranges  
- Leaderboard build succeeds  

CI does **not** recompute embeddings or retrain probes on maintainer infrastructure.

---

## Protocol summary

- Use the pinned dataset revision and harness settings from `benchmark_spec.json`.  
- Do not train on the test split or perform encoder fine-tuning on task labels beyond the
  defined linear probe.  
- One submission file per model; one pull request per model is easiest to review.  
