#!/usr/bin/env python3
"""
Build the two leaderboard files the HF Space serves, from the reviewed submissions.

    submissions/<model_id>.json  +  benchmark/benchmark_spec.json  +  benchmark/model_meta.json
        -> leaderboard/leaderboard.json        (13 task-type aggregates)
        -> leaderboard/leaderboard_tasks.json  (100 per-task scores)

Aggregates metrics from reviewed submissions; does not run the evaluation harness.
Macro = mean of the 13 category means (each weighs 1/13).
Micro = mean over all 100 tasks.
"""
import json, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "benchmark", "benchmark_spec.json")))
META = json.load(open(os.path.join(ROOT, "benchmark", "model_meta.json")))

CAT_ORDER = SPEC["category_order"]
TASKS     = SPEC["tasks"]                       # [{id,label,category}] x100
TASK_CAT  = {t["id"]: t["category"] for t in TASKS}
TASK_IDS  = [t["id"] for t in TASKS]            # already ordered by category
REGIMES   = SPEC["regimes"]
METRICS   = SPEC["metrics"]                     # ["MCC","Acc","F1"]
R4        = lambda x: round(float(x), 4)


def load_submissions():
    subs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "submissions", "*.json"))):
        subs.append(json.load(open(path)))
    return subs


def aggregate(per_task, metric):
    """per_task: {task_id: {MCC,Acc,F1}} -> (micro, macro, {cat: mean})."""
    vals = {tid: per_task[tid][metric] for tid in TASK_IDS}
    cat_mean = {}
    for c in CAT_ORDER:
        xs = [vals[t] for t in TASK_IDS if TASK_CAT[t] == c]
        cat_mean[c] = sum(xs) / len(xs)
    micro = sum(vals.values()) / len(vals)
    macro = sum(cat_mean.values()) / len(CAT_ORDER)
    return micro, macro, cat_mean


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
                micro, macro, cmean = aggregate(per, mk)
                crow[f"micro_{mk}"] = R4(micro)
                crow[f"macro_{mk}"] = R4(macro)
                crow[f"cat_{mk}"]   = {c: R4(cmean[c]) for c in CAT_ORDER}
                trow[f"micro_{mk}"] = R4(micro)
                trow[f"macro_{mk}"] = R4(macro)
                trow[f"task_{mk}"]  = {t: R4(per[t][mk]) for t in TASK_IDS}
            cat_data[reg].append(crow)
            task_data[reg].append(trow)

    for reg in REGIMES:                          # stable ordering for clean diffs
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
