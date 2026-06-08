#!/usr/bin/env python3
"""
Upload or download the pinned GENEB task CSVs on Hugging Face.

HF layout (Dataset Viewer): one subset per task (config_name = safe slug of label),
each task folder has train.csv + test.csv (real HF splits, no split column in files).

Local harness layout: flat GENEB_data/<task_id>.csv with columns text,label,split.

Requires: python3 -m pip install "huggingface_hub>=0.24"
Auth: huggingface-cli login  OR  export HF_TOKEN=hf_...

  python3 tools/sync_geneb_dataset.py upload --data_dir ./GENEB_data
  python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "benchmark", "benchmark_spec.json")
DEFAULT_REPO = "darlednik/geneb-tasks"
DEFAULT_TAR = os.path.join(ROOT, "data_dir_all_csvs.tar.gz")
HF_TASKS_DIR = "tasks"
PLACEHOLDER_REV = "<PIN_DATASET_COMMIT_SHA>"
SPLIT_COLS = ("text", "label")


def load_spec():
    return json.load(open(SPEC_PATH))


def save_spec(spec):
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


def task_records(spec):
    return spec["tasks"]


def task_ids(spec):
    return [t["id"] for t in spec["tasks"]]


def config_slug(label: str) -> str:
    """HF config_name: no spaces; must be unique (task labels are unique)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return s.strip("_") or "task"


def yaml_quote(s: str) -> str:
    if re.search(r'[:#\[\]{},&*!|>\'"%@`]', s) or " " in s or s.startswith(("-", "?")):
        return json.dumps(s, ensure_ascii=False)
    return s


def ensure_local_csvs(data_dir: str, spec) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tasks = task_ids(spec)
    present = {t for t in tasks if os.path.isfile(os.path.join(data_dir, t + ".csv"))}
    if len(present) == len(tasks):
        return
    if not os.path.isfile(DEFAULT_TAR):
        raise SystemExit(
            f"ERROR: {data_dir} has {len(present)}/100 CSVs and {DEFAULT_TAR} is missing."
        )
    print(f"Extracting {DEFAULT_TAR} -> {data_dir} ...")
    with tarfile.open(DEFAULT_TAR, "r:gz") as tf:
        tf.extractall(data_dir)
    present = {t for t in tasks if os.path.isfile(os.path.join(data_dir, t + ".csv"))}
    if len(present) != len(tasks):
        missing = sorted(set(tasks) - present)[:5]
        raise SystemExit(f"ERROR: after extract still missing CSV(s), e.g. {missing}")


def check_local_flat(data_dir: str, spec) -> None:
    missing = [t for t in task_ids(spec) if not os.path.isfile(os.path.join(data_dir, t + ".csv"))]
    if missing:
        raise SystemExit(f"ERROR: missing {len(missing)} flat CSV(s), e.g. {missing[:3]}")


def split_task_csv(src: str, task_hf_dir: str) -> tuple[int, int]:
    """Write tasks/<id>/train.csv and test.csv with columns text,label only."""
    os.makedirs(task_hf_dir, exist_ok=True)
    rows = list(csv.DictReader(open(src)))
    need = {"text", "label", "split"}
    if not rows or not need.issubset(rows[0].keys()):
        raise ValueError(f"{src}: need columns text,label,split")

    buckets = {"train": [], "test": []}
    skipped = set()
    for r in rows:
        sp = (r.get("split") or "").strip()
        if sp not in buckets:
            skipped.add(sp)  # e.g. dev — not used by GENEB harness
            continue
        buckets[sp].append({k: r[k] for k in SPLIT_COLS})

    for sp, part in buckets.items():
        out = os.path.join(task_hf_dir, f"{sp}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(SPLIT_COLS))
            w.writeheader()
            w.writerows(part)
    return skipped


def build_readme_markdown_body(spec) -> list[str]:
    """Human-readable dataset card text (shown on the Hub above the task table)."""
    n_tasks = len(spec["tasks"])
    n_cats = len(spec["category_order"])
    regimes = ", ".join(spec["regimes"])
    metrics = ", ".join(spec["metrics"])
    seeds = ", ".join(str(s) for s in spec["protocol"]["seeds"])

    lines = [
        "# GENEB — Genomic Embedding Benchmark (task data)",
        "",
        "Task-level sequence classification data for "
        "[GENEB](https://github.com/darlednik/geneb): a multi-task benchmark for DNA sequence "
        "encoders evaluated with **precomputed representations and logistic regression** "
        f"(no encoder fine-tuning on task labels) across **{n_tasks} tasks** in "
        f"**{n_cats} functional categories**.",
        "",
        "This repository release contains **train and test partitions only**. Model scores, "
        "evaluation code, and submission artifacts are maintained in the "
        "[benchmark repository](https://github.com/darlednik/geneb) and summarized on the "
        "[leaderboard](https://huggingface.co/spaces/darlednik/geneb-leaderboard).",
        "",
        "## Record schema",
        "",
        "Each example is a single row with:",
        "",
        "- **`text`** — nucleotide sequence (DNA over {A,C,G,T}; length and windowing are defined by the source task).",
        "- **`label`** — discrete class label for the assay or prediction target.",
        "",
        "Train and test examples are provided in separate files per task (`train.csv`, `test.csv`). "
        "Auxiliary splits present in some upstream releases (e.g. `dev`, `validation`) are "
        "**excluded** from this distribution and are not used in the GENEB evaluation protocol.",
        "",
        "## Evaluation protocol (summary)",
        "",
        "GENEB scores models on these tasks under a shared protocol (full specification in the "
        "[benchmark repository](https://github.com/darlednik/geneb/blob/main/benchmark/benchmark_spec.json)):",
        "",
        f"| Item | Specification |",
        f"|------|----------------|",
        f"| Tasks | {n_tasks} |",
        f"| Functional categories | {n_cats} |",
        f"| Reporting regimes | {regimes} |",
        f"| Metrics | {metrics} (primary: **{spec.get('primary_metric', 'MCC')}**) |",
        f"| Probe | {spec['protocol']['probe']} (`max_iter={spec['protocol']['logreg']['max_iter']}`) |",
        f"| Random seeds (mean over runs) | {seeds} |",
        "",
        "Participants should use the released train/test assignments as-is; relabeling, "
        "resplitting, or training on the test partition is outside the protocol.",
        "",
        "## Task categories",
        "",
        "| Category | Tasks |",
        "|----------|------:|",
    ]
    counts = spec.get("categories") or {}
    for cat in spec["category_order"]:
        lines.append(f"| {cat} | {counts.get(cat, 0)} |")
    lines.extend([
        "",
        "## Repository layout and Dataset Viewer",
        "",
        "Data are organized as `tasks/<task_id>/train.csv` and `tasks/<task_id>/test.csv` "
        "with columns `text` and `label`.",
        "",
        "On the Hub Dataset Viewer, select:",
        "",
        "1. **Subset** (`config_name`) — one benchmark task (short slug, e.g. `NT_H3`; see index below).",
        "2. **Split** — `train` or `test`.",
        "",
        "## Local download (harness)",
        "",
        "To reproduce runs with the reference harness (uses the dataset revision pinned in that spec):",
        "",
        "```bash",
        "python3 -m pip install \"huggingface_hub>=0.24\"",
        "python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data",
        "```",
        "",
        "This materializes `GENEB_data/<task_id>.csv` with columns `text`, `label`, and `split` "
        "for use with `harness/run_GENEB.py`.",
        "",
        "## Task index",
        "",
        "| Subset (`config_name`) | Short label | Task identifier | Category |",
        "|------------------------|-------------|-----------------|----------|",
    ])
    for task in task_records(spec):
        lines.append(
            f"| `{config_slug(task['label'])}` | {task['label']} | `{task['id']}` | {task['category']} |"
        )
    lines.append("")
    return lines


def build_hf_staging(flat_dir: str, staging_dir: str, spec) -> str:
    """tasks/<task_id>/{train,test}.csv + README.md with 100 HF configs."""
    tasks_root = os.path.join(staging_dir, HF_TASKS_DIR)
    if os.path.isdir(tasks_root):
        shutil.rmtree(tasks_root)
    os.makedirs(tasks_root, exist_ok=True)

    lines = [
        "---",
        "language:",
        "- en",
        "license: apache-2.0",
        "task_categories:",
        "- text-classification",
        "tags:",
        "- biology",
        "- genomics",
        "- dna",
        "- benchmark",
        "- dna-language-model",
        "configs:",
    ]

    skipped_splits = set()
    for i, task in enumerate(task_records(spec)):
        tid, label, cat = task["id"], task["label"], task["category"]
        src = os.path.join(flat_dir, tid + ".csv")
        task_hf = os.path.join(tasks_root, tid)
        skipped_splits |= split_task_csv(src, task_hf)

        slug = config_slug(label)
        cfg = f"- config_name: {slug}\n"
        if i == 0:
            cfg += "  default: true\n"
        cfg += f"  data_dir: {yaml_quote(f'{HF_TASKS_DIR}/{tid}')}\n"
        lines.append(cfg.rstrip())

    lines.append("---")
    lines.append("")
    lines.extend(build_readme_markdown_body(spec))

    readme = os.path.join(staging_dir, "README.md")
    with open(readme, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Built HF staging: {staging_dir} ({len(task_records(spec))} tasks, train+test each)")
    if skipped_splits:
        print(f"  (ignored extra split values: {sorted(skipped_splits)} — GENEB uses train/test only)")
    return readme


def merge_hf_tasks_to_flat(hf_root: str, flat_dir: str, spec) -> None:
    """Rebuild <task_id>.csv with text,label,split from tasks/<id>/{train,test}.csv."""
    os.makedirs(flat_dir, exist_ok=True)
    tasks_root = os.path.join(hf_root, HF_TASKS_DIR)
    if not os.path.isdir(tasks_root):
        raise SystemExit(f"ERROR: expected {tasks_root}/ in downloaded dataset")

    for task in task_records(spec):
        tid = task["id"]
        task_hf = os.path.join(tasks_root, tid)
        out = os.path.join(flat_dir, tid + ".csv")
        merged = []
        for sp in ("train", "test"):
            part = os.path.join(task_hf, sp + ".csv")
            if not os.path.isfile(part):
                raise SystemExit(f"ERROR: missing {part}")
            for row in csv.DictReader(open(part)):
                merged.append({**row, "split": sp})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "split"])
            w.writeheader()
            w.writerows(merged)


def hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit('Install: python3 -m pip install "huggingface_hub>=0.24"')
    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token) if token else HfApi()


def refresh_dataset_viewer(api, repo_id: str) -> None:
    """Parquet for the Viewer is rebuilt by HF (@parquet-converter), not via delete_branch."""
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        main_sha = next((b.target_commit for b in refs.branches if b.name == "main"), None)
        parquet_sha = next((c.target_commit for c in refs.converts if c.name == "parquet"), None)
        if main_sha and parquet_sha and main_sha == parquet_sha:
            print("Parquet conversion is up to date with main.")
            return
        if parquet_sha:
            print("Parquet conversion is stale (refs/convert/parquet != main).")
        print(
            "  HF will re-convert automatically — watch Discussions for @parquet-converter,\n"
            f"  then hard-refresh: https://huggingface.co/datasets/{repo_id}\n"
            "  (typically 5–30 min; cannot delete refs/convert/parquet via API.)"
        )
    except Exception as e:
        print(f"  (could not check parquet ref: {e})")


def legacy_root_csvs(api, repo_id: str) -> list[str]:
    """Paths of flat *.csv at repo root from the first upload layout."""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return []
    return [f for f in files if f.endswith(".csv") and "/" not in f]


def cmd_upload(args):
    spec = load_spec()
    data_dir = os.path.abspath(args.data_dir)
    repo_id = args.repo_id
    ensure_local_csvs(data_dir, spec)
    check_local_flat(data_dir, spec)

    api = hf_api()
    try:
        who = api.whoami()
    except Exception as e:
        raise SystemExit(
            "ERROR: Hugging Face token required.\n"
            "  python3 -c \"from huggingface_hub import login; login()\"\n"
            "  OR  export HF_TOKEN=hf_...\n"
            f"  ({e})"
        ) from e
    print(f"Logged in as: {who.get('name', who)}")

    staging = tempfile.mkdtemp(prefix="geneb_hf_")
    try:
        build_hf_staging(data_dir, staging, spec)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        to_delete = legacy_root_csvs(api, repo_id) if args.clean_legacy else []
        if to_delete:
            print(f"Removing {len(to_delete)} legacy flat CSV(s) at repo root (same commit as upload) ...")
        print(f"Uploading HF bundle -> {repo_id} ...")
        api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=args.message,
            delete_patterns=to_delete or None,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    info = api.repo_info(repo_id, repo_type="dataset")
    revision = info.sha
    if not revision:
        raise SystemExit("ERROR: could not read dataset revision SHA after upload")

    spec.setdefault("dataset", {})
    spec["dataset"]["repo_id"] = repo_id
    spec["dataset"]["source"] = f"https://huggingface.co/datasets/{repo_id}"
    spec["dataset"]["revision"] = revision
    spec["dataset"]["layout"] = (
        "HF: tasks/<task_id>/{train,test}.csv + README configs (subset=task label); "
        "harness: flat <task_id>.csv with text,label,split (built by download)"
    )
    save_spec(spec)

    if args.refresh_viewer:
        refresh_dataset_viewer(api, repo_id)

    print(f"OK: pinned revision {revision}")
    print(f"     Viewer: Subset = task slug (see README table), Split = train | test")
    print(f"     https://huggingface.co/datasets/{repo_id}")
    if args.refresh_viewer:
        print("     Wait for Parquet reconversion before the Viewer updates.")
    print(f"Download: python3 tools/sync_geneb_dataset.py download --local_dir ./GENEB_data")


def cmd_repair_viewer(args):
    """Re-upload README (fixed configs) and clear stale Parquet — no CSV re-upload."""
    spec = load_spec()
    data_dir = os.path.abspath(args.data_dir)
    repo_id = args.repo_id
    check_local_flat(data_dir, spec)

    api = hf_api()
    try:
        who = api.whoami()
    except Exception as e:
        raise SystemExit(f"ERROR: HF token required ({e})") from e
    print(f"Logged in as: {who.get('name', who)}")

    staging = tempfile.mkdtemp(prefix="geneb_hf_")
    try:
        readme = build_hf_staging(data_dir, staging, spec)
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="fix: per-task subsets + train/test splits for Dataset Viewer",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    info = api.repo_info(repo_id, repo_type="dataset")
    revision = info.sha
    spec.setdefault("dataset", {})["revision"] = revision
    save_spec(spec)
    refresh_dataset_viewer(api, repo_id)
    print(f"OK: README + viewer refresh @ {revision[:12]}...")


def cmd_download(args):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit('Install: python3 -m pip install "huggingface_hub>=0.24"')

    spec = load_spec()
    ds = spec.get("dataset") or {}
    repo_id = args.repo_id or ds.get("repo_id")
    revision = args.revision or ds.get("revision")
    if not repo_id:
        raise SystemExit("ERROR: dataset.repo_id missing in benchmark_spec.json")
    if not revision or revision == PLACEHOLDER_REV:
        raise SystemExit("ERROR: dataset.revision not pinned. Run: upload first.")

    flat_dir = os.path.abspath(args.local_dir)
    cache = tempfile.mkdtemp(prefix="geneb_dl_")
    try:
        print(f"Downloading {repo_id} @ {revision[:12]}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=cache,
        )
        tasks_root = os.path.join(cache, HF_TASKS_DIR)
        if os.path.isdir(tasks_root):
            merge_hf_tasks_to_flat(cache, flat_dir, spec)
        else:
            # fallback: flat <task_id>.csv at dataset repo root (legacy layout)
            for tid in task_ids(spec):
                src = os.path.join(cache, tid + ".csv")
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(flat_dir, tid + ".csv"))
        check_local_flat(flat_dir, spec)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"OK: {len(task_ids(spec))} flat CSVs in {flat_dir}")


def main():
    ap = argparse.ArgumentParser(description="Upload/download GENEB pinned dataset on HF")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload HF-friendly layout; pin revision in spec")
    up.add_argument("--data_dir", default=os.path.join(ROOT, "GENEB_data"))
    up.add_argument("--repo_id", default=DEFAULT_REPO)
    up.add_argument("--message", default="GENEB: 100 tasks as subsets with train/test splits")
    up.add_argument("--clean-legacy", action="store_true", default=True,
                    help="delete old flat *.csv at repo root (default: on)")
    up.add_argument("--no-clean-legacy", action="store_false", dest="clean_legacy")
    up.add_argument("--refresh-viewer", action="store_true", default=True,
                    help="delete stale refs/convert/parquet after upload (default: on)")
    up.add_argument("--no-refresh-viewer", action="store_false", dest="refresh_viewer")
    up.set_defaults(func=cmd_upload)

    fix = sub.add_parser(
        "repair-viewer",
        help="Fix Dataset Viewer dropdowns (README configs + clear stale Parquet)",
    )
    fix.add_argument("--data_dir", default=os.path.join(ROOT, "GENEB_data"))
    fix.add_argument("--repo_id", default=DEFAULT_REPO)
    fix.set_defaults(func=cmd_repair_viewer)

    down = sub.add_parser("download", help="Download pinned revision -> flat GENEB_data/")
    down.add_argument("--local_dir", default=os.path.join(ROOT, "GENEB_data"))
    down.add_argument("--repo_id", default=None)
    down.add_argument("--revision", default=None)
    down.set_defaults(func=cmd_download)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
