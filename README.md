# GENEB — Genomic Embedding Benchmark

A community benchmark for **DNA foundation models**, evaluated as frozen embeddings + a
linear probe across **100 tasks** in **13 functional categories**, three regimes
(full-shot · 10-shot · 1-shot) and three metrics (MCC · Accuracy · macro-F1).

- **Live leaderboard:** https://huggingface.co/spaces/DeepPavlov/geneb-leaderboard
- **Add your model:** see [CONTRIBUTING.md](CONTRIBUTING.md)

---

## What this repo is

Two things in one place:

1. **The evaluation harness** (`harness/`) — the exact code that turns a model into
   embeddings and trains the linear probe. Contributors add their model **extractor** here — one small class implementing a single
   interface. No model extractors are bundled in this repo; the leaderboard's rows come from
   submitted metrics, and each contributor ships their own extractor with their PR.
2. **The benchmark + leaderboard pipeline** — a frozen task spec, the reviewed
   submissions, and the automation that rebuilds the public leaderboard and pushes it to
   the Hugging Face Space.

**We do not re-score submissions.** Integrity comes from *reproducibility*, not from a
central scorer: a fixed dataset revision, a pinned harness, fixed seeds and probe
settings, a standard result format, public PR review, and an honest provenance badge.
Anyone can re-run any model because the extractor is in the repo.

---

## Repository layout

```
benchmark/
  benchmark_spec.json     # frozen contract: 100 tasks, task->category, regimes, metrics, seeds, probe settings
  model_meta.json         # curated model registry (display name, params, links, provenance)
submissions/
  <model_id>.json         # one reviewed result file per model  <- contributors add here
leaderboard/
  index.html              # the static HF Space app
  leaderboard.json        # built: 13 task-type aggregates       (do not hand-edit)
  leaderboard_tasks.json  # built: 100 per-task scores           (do not hand-edit)
  README.md               # HF Space frontmatter
tools/
  pack_submission.py      # raw harness output  -> submissions/<model_id>.json
  build_leaderboard.py    # submissions/*       -> leaderboard/*.json
  validate_submission.py  # CI schema / completeness / range checks
  sync_geneb_dataset.py   # download/upload pinned HF dataset (GENEB_data/)
model_cards/
  <model_id>.md           # provenance & training-data disclosure (one per model)
harness/
  run_GENEB.py            # THE contributor command: extractor -> complete submission.json
  extractors/
    base.py               # the one interface every model implements
    <your_model>.py       # contributors add their extractor here (none bundled)
.github/workflows/
  validate.yml            # runs on every PR
  build-and-sync.yml      # runs on merge to main -> rebuild + push to the Space
```

> **Grafting onto the existing code:** keep the current `embedding_pipeline/` as `harness/`
> (only import paths change). Everything above is additive — no harness logic is touched.

---

## How a result becomes a leaderboard row

```
contributor writes extractor -> python harness/run_GENEB.py  (1 command, all 100 tasks)
        |                                                                   |
        v                                                                   v
 PR adds submissions/<id>.json + model_card                       validate.yml (CI checks schema)
        |                                                                   |
        v                                                                   v
 maintainer reviews & merges ----------------------------> build-and-sync.yml rebuilds
                                                            leaderboard JSON and pushes to the Space
```

The two leaderboard files are **derived artifacts** — never edited by hand. The single
source of truth is `submissions/` + `benchmark/`.

---

## Integrity model (no central scoring)

| Mechanism | What it guarantees |
|---|---|
| Pinned dataset revision | everyone trains/tests on identical splits |
| Pinned harness + seeds + probe settings | the probe is the same for everyone |
| Extractor committed with the submission | any third party can reproduce the numbers |
| CI validation | every task present, metrics in range, schema correct |
| Provenance badge | `self-reported` for community PRs, `official` for authors' own runs |
| Disclosure fields | zero-shot / contamination / training data stated up front |

Because the splits are public (we don't re-score), the benchmark is **self-reported by
design** and honest about it — the badge says so, and the extractor makes the claim checkable.

---

## Citation

```bibtex
@article{GENEB_dna_survey,
  title  = {DNA Survey: Benchmarking DNA Foundation Models},
  author = {DeepPavlov and contributors},
  year   = {2026}
}
```
