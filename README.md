# GENEB — Genomic Embedding Benchmark

[![ICML 2026](https://img.shields.io/badge/ICML-2026-1f6feb.svg)](https://arxiv.org/abs/2606.04525)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04525-b31b1b.svg)](https://arxiv.org/abs/2606.04525)
[![Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97-Leaderboard-yellow.svg)](https://huggingface.co/spaces/darlednik/geneb-leaderboard)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Task%20data-yellow.svg)](https://huggingface.co/datasets/darlednik/geneb-tasks)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**40 genomic foundation models. 100 DNA classification tasks. 13 functional categories.
One unified probing protocol.**

Progress in genomic foundation models is hard to assess: models are evaluated on disjoint
benchmarks under incompatible protocols, so claims of superiority are rarely comparable.
GENEB evaluates frozen representations from 40 models on the full 100-task suite with a
single linear-probe protocol, in full-, 10-shot, and 1-shot regimes — the role MTEB plays
in NLP, for genomics.

📄 [**Paper: *GENEB: Why Genomic Models Are Hard to Compare*** (ICML 2026)](https://arxiv.org/abs/2606.04525) ·
🏆 [**Interactive leaderboard**](https://huggingface.co/spaces/darlednik/geneb-leaderboard) ·
🧬 [**Task data**](https://huggingface.co/datasets/darlednik/geneb-tasks)

<img width="6961" height="2810" alt="image" src="https://github.com/user-attachments/assets/5fdaefcd-bcf8-4e3f-ab79-c91acc3fd27c" />

---

## Key findings

**1. Aggregate leaderboards are unstable.** Model rankings vary sharply across task
categories, so a single headline number hides which model you should actually use. Within the
top 6 by macro MCC, GenomeOcean-4B ranks 4th overall yet is the strongest of that group on
histone modification (0.545), and NT-2.5B-MS ranks 5th yet is the strongest on splice sites
(0.652).

**2. Scale gives only modest and inconsistent gains.** Parameter count is a weak predictor of
representation quality: neither 7B model in the suite reaches the top 6, and the largest model
that does — GenomeOcean-4B, macro 0.573 — scores below both the 3B and the 1.2B
GENERator-Eukaryote models (0.605 / 0.581).

**3. Architecture and pretraining alignment often outweigh parameter count.** Under matched
comparison across architecture, tokenization, and pretraining data, these factors frequently
matter more for representation quality than model size.

**4. Category difficulty is highly uneven.** Across those same six models, promoter tasks reach
0.739–0.774 MCC while DNA methylation stays at 0.231–0.440 — collapsing the two into one score
discards the information a practitioner needs.

*(Full-shot MCC. See the [paper](https://arxiv.org/abs/2606.04525) for per-category scaling
correlations, non-linear probe checks, and few-shot sensitivity analyses.)*

---

## Leaderboard — top 6 of 40

Full-shot, MCC. **Macro** = mean of the 13 category scores (each weighs 1/13); **Micro** =
mean over all 100 tasks.

| # | Model | Params | Macro | Micro |
|---|-------|--------|-------|-------|
| 1 | GENERator-eukaryote-3b | 3B | **0.605** | 0.598 |
| 2 | LucaOne-default-step36M | 2B | 0.589 | 0.579 |
| 3 | GENERator-eukaryote-1.2b | 1.2B | 0.581 | 0.581 |
| 4 | GenomeOcean-4B | 4B | 0.573 | 0.563 |
| 5 | NT-2.5B-MS | 2.5B | 0.573 | 0.561 |
| 6 | Omni-DNA-1B | 1B | 0.568 | 0.560 |

The same six models across 6 of the 13 categories — note how the ordering changes column to
column. Numbers in headers are task counts; **bold** marks the best of these six models in a
column, not the best of all 40.

| Model | Histone mod. (30) | Promoters (22) | Enhancers (8) | DNA methyl. (8) | Splice sites (7) | lncRNA (6) |
|-------|------|------|------|------|------|------|
| GENERator-eukaryote-3b | 0.537 | **0.774** | **0.488** | **0.440** | 0.648 | 0.453 |
| LucaOne-default-step36M | 0.533 | 0.746 | 0.479 | 0.323 | 0.636 | **0.508** |
| GENERator-eukaryote-1.2b | 0.521 | 0.768 | 0.485 | 0.397 | 0.629 | 0.438 |
| GenomeOcean-4B | **0.545** | 0.739 | **0.488** | 0.231 | 0.580 | 0.396 |
| NT-2.5B-MS | 0.498 | 0.739 | 0.487 | 0.356 | **0.652** | 0.372 |
| Omni-DNA-1B | 0.505 | 0.759 | 0.474 | 0.290 | 0.604 | 0.323 |

**[→ All 40 models, all 13 categories, all 100 tasks, in 1-/10-/full-shot](https://huggingface.co/spaces/darlednik/geneb-leaderboard)**
— filter by regime, metric (MCC / accuracy / macro-F1), and model size.

---

## Evaluate your model

Evaluation runs on your hardware. Model weights and intermediate embeddings stay local; a
submission contributes only the extractor code, the final metrics, and a model card.

```bash
# 1. Get the pinned task data revision
python3 -m pip install "huggingface_hub>=0.24"
python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data

# 2. Run the harness (start with --limit 5 for a smoke test)
python3 harness/run_GENEB.py \
  --extractor MyModelExtractor --module my_model \
  --name_model org/my-model \
  --model_id my-model-300m --display "My Model 300M" --params 300000000 \
  --data_dir ./GENEB_data --device cuda --limit 5

# 3. Validate the produced submission file
python3 tools/validate_submission.py submissions/my-model-300m.json
```

An extractor is a small wrapper that loads your encoder, prepares DNA sequences, computes
hidden states, applies a pooling strategy, and returns one embedding vector per sequence.
Full instructions: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Protocol

| | |
|---|---|
| Models evaluated | 40 genomic foundation models |
| Tasks | 100 DNA classification tasks in 13 functional categories |
| Data format | one CSV per task with columns `text`, `label`, `split` (`train` / `test`) |
| Evaluation | logistic-regression probe on frozen embeddings (`max_iter=1000`) |
| Regimes | `full`, `10shot`, `1shot` |
| Metrics | MCC (primary; robust to class imbalance), accuracy, macro-F1 |
| Seeds | `13, 17, 42, 123, 997` — results averaged over all five |
| Subsampling | tasks exceeding 10⁵ sequences are subsampled; MCC was shown empirically to stabilize beyond this size |
| Data pin | dataset revision fixed in `benchmark/benchmark_spec.json` |

Frozen representations isolate representation quality and keep comparisons controlled across
architectures and training regimes. Released train/test partitions are used as published,
without relabeling or resplitting; encoders are not fine-tuned on task labels beyond the
defined linear probe.

The authors declare no financial conflicts of interest: none of the 40 evaluated models was
developed by the authors or their funders.

---

## Models covered

<details>
<summary><b>All 40 evaluated models</b> (architecture · tokenization · params · pretraining data)</summary>

| Model | Architecture | Tokenization | Params | Pretraining data |
|-------|--------------|--------------|--------|------------------|
| METAGENE-1 | T-dec | BPE | 7B | multi-species |
| Evo-1-131k | StripedHyena | SN | 7B | prokaryotic |
| GenomeOcean-4B | T-dec | BPE | 4B | multi-species |
| GENERator-Eukaryote-3B | T-dec | k-mer | 3B | eukaryotic-genes |
| DNA-GPT-3B-M | T-dec | k-mer | 3B | multi-species |
| NT-2.5B-MS | T-enc | k-mer | 2.5B | multi-species |
| LucaOne | T-enc | mixed | 2B | multi-species |
| GENERator-Eukaryote-1.2B | T-dec | k-mer | 1.2B | eukaryotic-genes |
| Omni-DNA-1B | T-dec | BPE | 1B | multi-species |
| Agro-NT-1B | T-enc | k-mer | 1B | plant-genomes |
| SPACE | CNN-Transformer-MoE | SN | 589M | human-mouse-profiles |
| eccDNAMamba | Mamba | BPE | 537M | multi-species |
| GenomeOcean-500M | T-dec | BPE | 500M | multi-species |
| GENA-LM-Large-T2T | T-enc | BPE | 336M | multi-species |
| Omni-DNA-300M | T-dec | BPE | 300M | multi-species |
| BioFM-265M | T-dec | BioToken | 265M | human |
| Enformer | CNN-Transformer | SN | 252M | human-mouse-profiles |
| NT-v2-250M-MS | T-enc | k-mer | 250M | multi-species |
| PlantCaduceus | Mamba | SN | 225M | plant-genomes |
| OmniNA-220M | T-dec | BPE | 220M | multi-species |
| GPT2-Gene-Multi-v2 | T-dec | BPE | 200M | human |
| GPT2-Gene-v1 | T-dec | BPE | 200M | human |
| Genomics-FM | T-enc | BPE+k-mer | 120M | multi-species |
| DNABERT-S | T-enc | k-mer | 117M | multi-species-microbial |
| DNABERT-2 | T-enc | BPE | 117M | multi-species |
| GENA-LM-T2T-Multi | T-enc | BPE | 110M | multi-species |
| GENA-LM | T-enc | BPE | 110M | human |
| DNA-GPT-0.1B-H | T-dec | k-mer | 100M | multi-species |
| NT-v2-100M-MS | T-enc | k-mer | 100M | multi-species |
| GROVER | T-enc | BPE | 87M | human |
| MutBERT | T-enc | SN | 86M | human |
| DeepGene | T-enc+Graph-Transformer | BPE | 85M | human |
| HyenaDNA-Large-1M | Hyena | SN | 55M | human |
| NT-v2-50M-3mer-MS | T-enc | k-mer | 50M | multi-species |
| NT-v2-50M-MS | T-enc | k-mer | 50M | multi-species |
| HyenaDNA-Medium-160k | Hyena | SN | 14M | human |
| Caduceus-PS-131k | Mamba | SN | 8M | human |
| JanusDNA-72-w | Hybrid-Mamba-MoE | SN | 2M | human |
| JanusDNA-72-wo | Hybrid-Mamba-MoE | SN | 2M | human |
| Caduceus-PH-1k | Mamba | SN | 2M | human |

</details>

---

## Repository overview

Three coordinated parts:

1. **Evaluation harness** (`harness/`) — computes embeddings, trains the protocol-defined
   logistic-regression probe, and emits a submission file.
2. **Benchmark definition** (`benchmark/`) — task list, category map, probe settings, and
   model registry metadata.
3. **Leaderboard artifacts** (`leaderboard/`) — static site and JSON tables built by CI from
   reviewed submissions; not edited manually.

Baseline and contributed results live in `submissions/<model_id>.json`. The repository stores
extractor code and metrics, but not third-party model weights.

<details>
<summary><b>Repository layout</b></summary>

```text
benchmark/
  benchmark_spec.json       # task list, categories, metrics, probe protocol, dataset pin
  model_meta.json           # display names, parameter counts, links, provenance labels
submissions/
  <model_id>.json           # per-model results (reviewed before merge)
leaderboard/
  index.html                # Hugging Face Space UI
  leaderboard.json          # macro scores by functional category (generated)
  leaderboard_tasks.json    # per-task scores (generated)
  README.md                 # Space metadata
tools/
  validate_submission.py    # schema and completeness checks (CI)
  build_leaderboard.py      # aggregate submissions into leaderboard JSON
  pack_submission.py        # convert legacy per-task JSON logs to submission format
  sync_geneb_dataset.py     # download or publish pinned task CSVs on Hugging Face
model_cards/
  <model_id>.md             # optional training-data and disclosure notes
harness/
  run_GENEB.py              # end-to-end local evaluation → submission JSON
  extractors/
    base.py                 # extractor interface
    <module>.py             # model-specific embedding module (per submission)
.github/workflows/
  validate.yml              # PR checks on submissions and spec
  build-and-sync.yml        # rebuild leaderboard and push to the Space (on merge to main)
```

</details>

---

## From model to leaderboard row

A contributor implements an extractor under `harness/extractors/`, runs
`harness/run_GENEB.py` on the pinned task data, and opens a pull request with the submission
file, the extractor module, and a model card. CI validates schema and completeness;
maintainers review the extractor and metadata for plausibility; on merge, the leaderboard is
rebuilt and synced to the Hugging Face Space automatically.

---

## Evaluation and reproducibility

GENEB does **not** provide a central re-scoring service. Contributors run evaluation locally
using the published harness and the pinned dataset revision. Maintainers review pull requests
for schema compliance, completeness, and plausibility, but do not re-run every model on
maintainer infrastructure. Community submissions are marked `self-reported` in metadata.

| Mechanism | Role |
|-----------|------|
| Dataset revision in `benchmark_spec.json` | Fixes the exact benchmark data used for evaluation |
| Fixed probe protocol | Keeps downstream evaluation comparable across models |
| Submission schema + CI validation | Checks completeness, metric ranges, and file consistency |
| Extractor code in the PR | Shows how embeddings were produced and helps others reproduce the run |
| Model card | Documents architecture, training data, and possible benchmark overlap |
| Provenance metadata | Marks externally submitted runs as self-reported |

A run is considered reproducible when the submitted extractor, the GENEB harness version, and
the pinned dataset revision are sufficient for another user to repeat the evaluation.

---

## Citation

Accepted at the **43rd International Conference on Machine Learning (ICML 2026)**, Seoul,
South Korea. Please cite the conference version:

```bibtex
@inproceedings{ledneva2026geneb,
  title     = {{GENEB}: Why Genomic Models Are Hard to Compare},
  author    = {Ledneva, Daria and Nuridinov, Mikhail and Kuznetsov, Denis},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  note      = {To appear}
}
```

<details>
<summary>arXiv preprint entry</summary>

```bibtex
@misc{ledneva2026geneb-arxiv,
  title         = {{GENEB}: Why Genomic Models Are Hard to Compare},
  author        = {Ledneva, Daria and Nuridinov, Mikhail and Kuznetsov, Denis},
  year          = {2026},
  eprint        = {2606.04525},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.04525}
}
```

</details>

<!-- TODO: once PMLR publishes the volume, drop `note = {To appear}` and add `pages` + `url`. -->

---

## License

Code in this repository is released under the [Apache License 2.0](LICENSE).

The license covers the harness, tools, benchmark definition, and leaderboard code. It does
**not** cover third-party model weights, and it does not override the licensing terms of the
upstream sources from which GENEB tasks are derived — see the
[task data card](https://huggingface.co/datasets/darlednik/geneb-tasks) for per-source terms.

---

## Links

- **Paper:** https://arxiv.org/abs/2606.04525
- **Leaderboard:** https://huggingface.co/spaces/darlednik/geneb-leaderboard
- **Task data:** https://huggingface.co/datasets/darlednik/geneb-tasks
- **Submit a model:** [CONTRIBUTING.md](CONTRIBUTING.md)

Reference extractors for additional leaderboard models are currently on the
[`dev`](https://github.com/darlednik/geneb/tree/dev) branch and are being integrated into
`main`; `main` currently ships a minimal reference set under `harness/extractors/`.

## Contact

Repository and leaderboard contact: [Daria Ledneva](mailto:a.ledn2026@gmail.com).
Questions, suggestions, feedback, and model submissions are welcome.
