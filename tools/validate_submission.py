#!/usr/bin/env python3
"""
Validate one or more submission files against the benchmark spec. Used by CI on every PR.

    python tools/validate_submission.py                      # validate all submissions/*.json
    python tools/validate_submission.py submissions/Foo.json # validate one

Exits non-zero on failure. Schema and completeness checks only; metrics are not recomputed.
"""
import json, sys, os, glob, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "benchmark", "benchmark_spec.json")))
META = json.load(open(os.path.join(ROOT, "benchmark", "model_meta.json")))
OFFICIAL_IDS = {m for m, v in META.items() if v.get("provenance") == "official"}
TASK_IDS = {t["id"] for t in SPEC["tasks"]}
REGIMES  = set(SPEC["regimes"])
METRICS  = SPEC["metrics"]
PROVENANCE_ALLOWED = {"official", "verified", "reproduced", "self-reported"}
RANGES = {"MCC": (-1.0, 1.0), "Acc": (0.0, 1.0), "F1": (0.0, 1.0)}
SAFE = re.compile(r"[^A-Za-z0-9._-]")


def validate(path):
    errs = []
    name = os.path.basename(path)
    try:
        s = json.load(open(path))
    except Exception as e:
        return [f"{name}: not valid JSON ({e})"]

    mid = s.get("model_id")
    if not mid:
        errs.append(f"{name}: missing 'model_id'")
    elif SAFE.sub("_", mid) + ".json" != name:
        errs.append(f"{name}: filename must equal sanitized model_id + .json "
                    f"(expected {SAFE.sub('_', mid)}.json)")
    if not s.get("harness_version"):
        errs.append(f"{name}: missing 'harness_version' (pin the harness you ran)")

    meta = s.get("meta") or {}
    if not meta.get("display"):
        errs.append(f"{name}: meta.display is required")
    p = meta.get("params")
    if not isinstance(p, int) or p < 0:
        errs.append(f"{name}: meta.params must be a non-negative integer (parameter count)")
    prov = meta.get("provenance", "self-reported")
    if prov not in PROVENANCE_ALLOWED:
        errs.append(f"{name}: meta.provenance must be one of {sorted(PROVENANCE_ALLOWED)}")
    if prov == "official" and mid not in OFFICIAL_IDS:
        errs.append(f"{name}: provenance 'official' is reserved for the benchmark authors' "
                    f"curated models; external submissions must use 'self-reported'")
    if not isinstance(meta.get("zero_shot", False), bool):
        errs.append(f"{name}: meta.zero_shot must be true/false")

    results = s.get("results") or {}
    missing_reg = REGIMES - set(results)
    if missing_reg:
        errs.append(f"{name}: missing regimes {sorted(missing_reg)} (need {sorted(REGIMES)})")

    for reg in REGIMES & set(results):
        per = results[reg]
        got = set(per)
        if got != TASK_IDS:
            miss, extra = TASK_IDS - got, got - TASK_IDS
            if miss:  errs.append(f"{name}/{reg}: missing {len(miss)} task(s), e.g. {sorted(miss)[:3]}")
            if extra: errs.append(f"{name}/{reg}: unknown task(s), e.g. {sorted(extra)[:3]}")
        for tid, scores in per.items():
            for mk in METRICS:
                if mk not in scores:
                    errs.append(f"{name}/{reg}/{tid}: missing metric {mk}"); continue
                v = scores[mk]
                if not isinstance(v, (int, float)):
                    errs.append(f"{name}/{reg}/{tid}/{mk}: not a number"); continue
                lo, hi = RANGES[mk]
                if not (lo - 1e-6 <= v <= hi + 1e-6):
                    errs.append(f"{name}/{reg}/{tid}/{mk}: {v} out of range [{lo},{hi}]")
    return errs


def main(argv):
    paths = argv[1:] or sorted(glob.glob(os.path.join(ROOT, "submissions", "*.json")))
    all_errs = []
    for p in paths:
        all_errs += validate(p)
    if all_errs:
        print("VALIDATION FAILED:")
        for e in all_errs:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {len(paths)} submission(s) valid.")


if __name__ == "__main__":
    main(sys.argv)
