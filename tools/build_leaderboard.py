#!/usr/bin/env python3
"""
Build the two leaderboard files the HF Space serves, from the reviewed submissions.

    submissions/<model_id>.json  +  benchmark/benchmark_spec.json  +  benchmark/model_meta.json
        -> leaderboard/leaderboard.json        (13 task-type aggregates)
        -> leaderboard/leaderboard_tasks.json  (100 per-task scores)

Aggregates metrics from reviewed submissions; does not run the evaluation harness.
No rounding: category/micro/macro are plain means of per-task values from submissions.
Macro = mean of the 13 category means (each weighs 1/13).
Micro = mean over all 100 tasks.
"""
import json, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "benchmark", "benchmark_spec.json")))
META = json.load(open(os.path.join(ROOT, "benchmark", "model_meta.json")))

CAT_ORDER = SPEC["category_order"]
TASKS     = SPEC["tasks"]
TASK_CAT  = {t["id"]: t["category"] for t in TASKS}
TASK_IDS  = [t["id"] for t in TASKS]
REGIMES   = SPEC["regimes"]
METRICS   = SPEC["metrics"]


def load_submissions():
    subs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "submissions", "*.json"))):
        s = json.load(open(path))
        full = (s.get("results") or {}).get("full") or {}
        if len(full) != len(TASK_IDS):
            print(f"skip {os.path.basename(path)}: {len(full)}/{len(TASK_IDS)} tasks")
            continue
        subs.append(s)
    return subs


def aggregate(per_task, metric):
    """Means of raw per-task metrics (no per-task rounding before pooling)."""
    vals = {tid: float(per_task[tid][metric]) for tid in TASK_IDS}
    cat_scores = {}
    for c in CAT_ORDER:
        xs = [vals[t] for t in TASK_IDS if TASK_CAT[t] == c]
        cat_scores[c] = sum(xs) / len(xs)
    micro = sum(vals.values()) / len(vals)
    macro = sum(cat_scores.values()) / len(CAT_ORDER)
    return micro, macro, cat_scores


def build():
    subs = load_submissions()
    cat_data  = {r: [] for r in REGIMES}
    task_data = {r: [] for r in REGIMES}

    for s in subs:
        mid = s["model_id"]
        mm  = {**META.get(mid, {}), **s.get("meta", {})}
        disp   = mm.get("display", mid)
        params = mm.get("params", 0)
        prov   = mm.get("provenance", "self-reported")
        zshot  = bool(mm.get("zero_shot", False))

        for reg in REGIMES:
            per = s["results"][reg]
            crow = {"model": mid, "disp": disp, "params": params,
                    "official": prov == "official", "zero_shot": zshot, "submitted": prov}
            trow = {"model": mid, "disp": disp, "params": params}
            for mk in METRICS:
                micro, macro, cat_scores = aggregate(per, mk)
                crow[f"micro_{mk}"] = micro
                crow[f"macro_{mk}"] = macro
                crow[f"cat_{mk}"]   = cat_scores
                trow[f"micro_{mk}"] = micro
                trow[f"macro_{mk}"] = macro
                trow[f"task_{mk}"]  = {t: float(per[t][mk]) for t in TASK_IDS}
            cat_data[reg].append(crow)
            task_data[reg].append(trow)

    for reg in REGIMES:
        cat_data[reg].sort(key=lambda r: r["model"].lower())
        task_data[reg].sort(key=lambda r: r["model"].lower())

    out = os.path.join(ROOT, "leaderboard")
    os.makedirs(out, exist_ok=True)
    json.dump({"data": cat_data}, open(os.path.join(out, "leaderboard.json"), "w"),
              ensure_ascii=False, separators=(",", ":"))
    json.dump({"tasks": TASKS_META(), "data": task_data},
              open(os.path.join(out, "leaderboard_tasks.json"), "w"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"built leaderboard.json ({len(subs)} models) and leaderboard_tasks.json "
          f"({len(TASK_IDS)} tasks x {len(REGIMES)} regimes)")


def TASKS_META():
    return [{"id": t["id"], "label": t["label"], "cat": t["category"]} for t in TASKS]


if __name__ == "__main__":
    build()
