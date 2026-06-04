# Contributing a model to GENEB

The whole path is: **write one extractor class → run one command → open a PR.** You run the
evaluation yourself; you keep your weights and embeddings; nothing is re-scored centrally.
Your row shows up flagged `self-reported`.

---

## TL;DR

```bash
# 1. write harness/extractors/my_model.py  (one class, see below)
# 2. get the pinned dataset into ./GENEB_data  (one <task_id>.csv per task)
# 3. ONE command -> a complete, ready-to-PR submission:
python harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --url https://huggingface.co/org/my-model \
  --data_dir ./GENEB_data --device cuda --submitted_by "Your Name"
# -> submissions/my-model-300m.json   (all 100 tasks x 3 regimes, validated)
# 4. open a PR with that file + your extractor + a model card
```

---

## 1. Write an extractor

One class under `harness/extractors/<your_model>.py` that subclasses
`BaseEmbeddingExtractor` and returns one vector per sequence:

```python
from .base import BaseEmbeddingExtractor
import numpy as np

class MyModelExtractor(BaseEmbeddingExtractor):
    def __init__(self, name_model: str, device: str = "cpu"):
        super().__init__(name_model, device)
        # load tokenizer + model here

    def extract_embeddings(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        # return array of shape (len(sequences), hidden_size)
        ...
```

> **Note.** No model extractors are bundled in this repo — you bring your own (one file
> subclassing `BaseEmbeddingExtractor`). The leaderboard's existing rows come from the
> authors' submitted metrics; committing your extractor with your PR is what lets anyone
> reproduce your numbers — that's the benchmark's integrity model.

Tip: run on a couple of tasks first with `--limit 2` to check your extractor end-to-end
before launching the full run.

## 2. Get the pinned dataset

100 CSVs, each named exactly `<task_id>.csv` (ids in `benchmark/benchmark_spec.json`),
columns `text,label,split` (`split` in `{train,test}`). Pull the **pinned revision** from
Hugging Face so your splits match everyone else's. Don't reshuffle splits.

```bash
python3 -m pip install "huggingface_hub>=0.24"
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data
# uses dataset.repo_id + dataset.revision from benchmark/benchmark_spec.json
# (downloads tasks/<id>/train|test from HF and merges to flat CSVs for the harness)
```

On the Hub, each **subset** is one task (human-readable label); **train** / **test** are
real splits — not one table with a `split` column.

## 3. Run it — one command

```bash
python harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --data_dir ./GENEB_data --device cuda
```

`run_GENEB.py` checks the dataset is complete, then for **every** task extracts train/test
embeddings with your model and trains the frozen-embedding linear probe with the spec's
**fixed seeds `[13,17,42,123,997]`** and `LogisticRegression(max_iter=1000)`, in all three
regimes (full + few-shot k=1/k=10). It writes the means straight into
`submissions/<model_id>.json` in the canonical schema. You don't touch seeds, probe
settings, or metric definitions — that's what keeps every model comparable.

> Already ran the old `embedding_pipeline` and have a folder of `results_*.json`? Use
> `python tools/pack_submission.py --raw_dir ... --model_id ...` instead of step 3 — it
> folds that output into the same submission file.

## 4. Add a model card

Add `model_cards/<model_id>.md` (format in `model_cards/README.md`) and fill in architecture,
params, training data, and **any overlap with benchmark tasks** (contamination) or whether
results are **zero-shot**. Honesty here is the whole point of a self-reported benchmark.

## 5. Validate and open a PR

```bash
python tools/validate_submission.py submissions/my-model-300m.json
```

PR adds: `submissions/<id>.json`, `harness/extractors/<module>.py`, `model_cards/<id>.md`.
CI re-runs the checks; a maintainer reviews and merges; the leaderboard rebuilds and syncs
to the Space automatically. Your row appears flagged **`self-reported`**.

## What CI checks (and what it doesn't)

**Checks:** valid JSON; `model_id` matches filename; all 100 tasks present in each of 3
regimes; metrics in range (MCC ∈ [-1,1], Acc/F1 ∈ [0,1]); provenance `self-reported`; the
leaderboard still builds.

**Does not:** re-train the probe or recompute your metrics. The benchmark does not score for
you. Reproducibility — pinned data + pinned harness + your committed extractor — is the
guarantee.

## Rules of the road

- Use the pinned splits as-is — no training on test, no per-task fine-tuning beyond the
  standard frozen-embedding linear probe.
- Disclose zero-shot evaluation and any train/test contamination in the model card.
- One model per submission file; one PR per model is easiest to review.
