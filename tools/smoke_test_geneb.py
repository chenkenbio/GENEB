#!/usr/bin/env python3
"""
Smoke tests for GENEB harness + dataset layout.

  # 1) Compare two local data dirs (e.g. HF download vs ground-truth copy)
  python3 tools/smoke_test_geneb.py compare-dirs \\
      --a ./GENEB_data --b ./GENEB_data_real

  # 2) Download pinned HF data, then compare to reference dir
  python3 tools/smoke_test_geneb.py compare-hf \\
      --reference ./GENEB_data_real --local_dir ./GENEB_data

  # 3) Run k-mer extractor on a few tasks (end-to-end harness)
  python3 tools/smoke_test_geneb.py run-kmer \\
      --data_dir ./GENEB_data_real --limit 3

  # 4) Compare new run to old embedding_pipeline JSON outputs
  python3 tools/smoke_test_geneb.py compare-results \\
      --submission submissions/geneb-kmer-smoke.json \\
      --raw_dir /path/to/old/results --tolerance 1e-3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "benchmark", "benchmark_spec.json")


def load_spec():
    return json.load(open(SPEC_PATH))


def task_ids(spec):
    return [t["id"] for t in spec["tasks"]]


def split_counts(path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sp = (row.get("split") or "").strip()
            counts[sp] = counts.get(sp, 0) + 1
    return counts


def compare_dirs(a: str, b: str, spec) -> int:
    errs = []
    for tid in task_ids(spec):
        pa, pb = os.path.join(a, tid + ".csv"), os.path.join(b, tid + ".csv")
        if not os.path.isfile(pa):
            errs.append(f"missing in A: {tid}"); continue
        if not os.path.isfile(pb):
            errs.append(f"missing in B: {tid}"); continue
        ca, cb = split_counts(pa), split_counts(pb)
        for sp in ("train", "test"):
            if ca.get(sp) != cb.get(sp):
                errs.append(f"{tid}: {sp} rows  A={ca.get(sp)}  B={cb.get(sp)}")
                break
    if errs:
        print("COMPARE FAILED:")
        for e in errs[:20]:
            print(" ", e)
        if len(errs) > 20:
            print(f"  ... and {len(errs) - 20} more")
        return 1
    print(f"OK: {len(task_ids(spec))} tasks — train/test row counts match between")
    print(f"     {os.path.abspath(a)}")
    print(f"     {os.path.abspath(b)}")
    return 0


def compare_hf(reference: str, local_dir: str) -> int:
    print("Downloading pinned dataset ...")
    rc = subprocess.call(
        [sys.executable, os.path.join(ROOT, "tools", "sync_geneb_dataset.py"),
         "download", "--local_dir", local_dir],
        cwd=ROOT,
    )
    if rc != 0:
        return rc
    return compare_dirs(local_dir, reference, load_spec())


def run_kmer(data_dir: str, limit: int, out_id: str) -> int:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "harness", "run_GENEB.py"),
        "--extractor", "ExampleKmerExtractor",
        "--module", "example_kmer",
        "--name_model", "kmer-4",
        "--model_id", out_id,
        "--display", "GENEB k-mer smoke test",
        "--params", "256",
        "--data_dir", data_dir,
        "--device", "cpu",
        "--submitted_by", "smoke_test",
        "--limit", str(limit),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def _regime_of(fname: str) -> str:
    if fname.endswith("_k-1.json"):
        return "1shot"
    if fname.endswith("_k-10.json"):
        return "10shot"
    return "full"


def _task_of(fname: str, task_set: set[str]) -> str | None:
    best = None
    for tid in task_set:
        if tid in fname and (best is None or len(tid) > len(best)):
            best = tid
    return best


def compare_results(submission: str, raw_dir: str, tolerance: float) -> int:
    """Compare submission JSON to embedding_pipeline results_*.json means."""
    import glob

    spec = load_spec()
    task_set = set(task_ids(spec))
    sub = json.load(open(submission))
    got = sub.get("results") or {}

    raw_metrics: dict[tuple[str, str], dict] = {}
    for path in glob.glob(os.path.join(raw_dir, "results_*.json")):
        fname = os.path.basename(path)
        tid = _task_of(fname, task_set)
        if not tid:
            continue
        reg = _regime_of(fname)
        blob = json.load(open(path))
        raw_metrics[(tid, reg)] = {
            "MCC": blob["mcc"]["mean"],
            "Acc": blob["accuracy"]["mean"],
            "F1": blob["f1_score"]["mean"],
        }

    if not raw_metrics:
        print(f"ERROR: no results_*.json under {raw_dir}")
        return 1

    diffs = []
    for (tid, reg), old in sorted(raw_metrics.items()):
        new = (got.get(reg) or {}).get(tid)
        if not new:
            diffs.append(f"{tid}/{reg}: missing in submission"); continue
        for mk in ("MCC", "Acc", "F1"):
            a, b = float(old[mk]), float(new[mk])
            if abs(a - b) > tolerance:
                diffs.append(f"{tid}/{reg}/{mk}: old={a:.4f} new={b:.4f} delta={abs(a-b):.4f}")

    if diffs:
        print("METRIC DIFFS (tolerance", tolerance, "):")
        for d in diffs[:30]:
            print(" ", d)
        return 1
    print(f"OK: {len(raw_metrics)} task×regime blocks match within ±{tolerance}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare-dirs")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)

    h = sub.add_parser("compare-hf")
    h.add_argument("--reference", required=True)
    h.add_argument("--local_dir", default=os.path.join(ROOT, "GENEB_data"))

    k = sub.add_parser("run-kmer")
    k.add_argument("--data_dir", default=os.path.join(ROOT, "GENEB_data_real"))
    k.add_argument("--limit", type=int, default=2)
    k.add_argument("--model_id", default="geneb-kmer-smoke")

    r = sub.add_parser("compare-results")
    r.add_argument("--submission", required=True)
    r.add_argument("--raw_dir", required=True)
    r.add_argument("--tolerance", type=float, default=1e-3)

    args = ap.parse_args()
    spec = load_spec()

    if args.cmd == "compare-dirs":
        sys.exit(compare_dirs(os.path.abspath(args.a), os.path.abspath(args.b), spec))
    if args.cmd == "compare-hf":
        sys.exit(compare_hf(os.path.abspath(args.reference), os.path.abspath(args.local_dir)))
    if args.cmd == "run-kmer":
        rc = run_kmer(os.path.abspath(args.data_dir), args.limit, args.model_id)
        if rc == 0:
            out = os.path.join(ROOT, "submissions", args.model_id + ".json")
            print(f"wrote {out}")
            if args.limit <= 0 or args.limit >= len(task_ids(load_spec())):
                subprocess.call(
                    [sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"), out],
                    cwd=ROOT,
                )
            else:
                print(f"(skipped full validate: --limit {args.limit}; OK for smoke test)")
        sys.exit(rc)
    if args.cmd == "compare-results":
        sys.exit(compare_results(args.submission, args.raw_dir, args.tolerance))


if __name__ == "__main__":
    main()
