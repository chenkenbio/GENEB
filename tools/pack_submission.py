#!/usr/bin/env python3
"""
Pack legacy per-task harness logs into a single submission file.

Earlier runs may emit one JSON file per task and regime:
    results_<modelstem>_<task_id>_<extractor>.json          (full-shot)
    results_<modelstem>_<task_id>_<extractor>_k-1.json       (1-shot)
    results_<modelstem>_<task_id>_<extractor>_k-10.json      (10-shot)
each containing {"accuracy":{mean,std}, "f1_score":{mean,std}, "mcc":{mean,std}}.

This collapses a run into submissions/<model_id>.json, keeping only the means and
mapping names to the benchmark's canonical task ids / regimes / metrics.

    python tools/pack_submission.py \
        --raw_dir ./my_run/results \
        --model_id my-dna-model-300m \
        --display "My DNA Model 300M" \
        --params 300000000 \
        --url https://huggingface.co/me/my-dna-model-300m \
        --submitted_by "Jane Doe" \
        [--zero_shot] [--training_data "human GRCh38 + ..."]

Validate with tools/validate_submission.py before opening a pull request.
"""
import argparse, json, os, re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "benchmark", "benchmark_spec.json")))
TASK_IDS = [t["id"] for t in SPEC["tasks"]]
TASK_SET = set(TASK_IDS)
SAFE = re.compile(r"[^A-Za-z0-9._-]")
# raw JSON field -> submission metric key
MKEY = {"mcc": "MCC", "accuracy": "Acc", "f1_score": "F1"}


def regime_of(fname):
    if fname.endswith("_k-1.json"):  return "1shot"
    if fname.endswith("_k-10.json"): return "10shot"
    return "full"


def task_of(fname):
    """Find which canonical task id appears in the filename (longest match wins)."""
    hits = [t for t in TASK_IDS if t in fname]
    return max(hits, key=len) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="folder with the harness results_*.json files")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--display", required=True)
    ap.add_argument("--params", type=int, required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--submitted_by", default="")
    ap.add_argument("--training_data", default="")
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--harness_version", default=SPEC.get("harness_version", "GENEB-0.1.0"))
    a = ap.parse_args()

    results = {r: {} for r in SPEC["regimes"]}
    seen = {r: set() for r in SPEC["regimes"]}
    for path in glob.glob(os.path.join(a.raw_dir, "*.json")):
        f = os.path.basename(path)
        tid = task_of(f)
        if tid is None:
            print(f"  ! skip (no known task id in name): {f}"); continue
        reg = regime_of(f)
        raw = json.load(open(path))
        row = {}
        for hk, mk in MKEY.items():
            if hk not in raw or "mean" not in raw[hk]:
                print(f"  ! {f}: missing {hk}.mean"); continue
            row[mk] = float(raw[hk]["mean"])
        results[reg][tid] = row
        seen[reg].add(tid)

    for reg in SPEC["regimes"]:
        miss = TASK_SET - seen[reg]
        if miss:
            print(f"  WARNING: {reg} missing {len(miss)} task(s); submission will fail validation "
                  f"until complete. e.g. {sorted(miss)[:3]}")

    sub = {
        "schema_version": "1.0",
        "model_id": a.model_id,
        "harness_version": a.harness_version,
        "meta": {
            "display": a.display, "params": a.params, "url": a.url,
            "zero_shot": a.zero_shot, "training_data": a.training_data,
            "provenance": "self-reported", "submitted_by": a.submitted_by,
        },
        "results": {reg: {t: results[reg][t] for t in TASK_IDS if t in results[reg]}
                    for reg in SPEC["regimes"]},
    }
    out = os.path.join(ROOT, "submissions", SAFE.sub("_", a.model_id) + ".json")
    json.dump(sub, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}")
    print("validate: python tools/validate_submission.py {}".format(
        os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
