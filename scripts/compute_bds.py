#!/usr/bin/env python3
"""Compute BDS calibration-to-holdout summaries for Wan2.1 BSS."""

import argparse
import csv
import math
import random
from pathlib import Path


SPLITS = ["first_half_split", "alternating_split", "random_split_seed0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_root = resolve_path(args.output_root, repo_dir)
    gain_path = output_root / "metrics/same_compute_gain_long.csv"
    rows = read_gain_rows(gain_path)
    bds_rows, split_summary = compute_bds(rows, args.n_boot)
    write_outputs(output_root, bds_rows, split_summary)
    print(f"Wrote BDS tables under {output_root / 'tables'}")


def compute_bds(rows, n_boot):
    case_ids = sorted({row["case_id"] for row in rows})
    available_nfe = sorted({int(row["actual_nfe"]) for row in rows})
    all_tset = [nfe for nfe in [10, 20, 30, 40] if nfe in available_nfe]
    if 30 not in available_nfe and all_tset:
        all_note = "Middle High 30 NFE missing; BDS_all uses available points."
    else:
        all_note = ""
    tsets = {
        "BDS_low": [10] if 10 in available_nfe else [],
        "BDS_all": all_tset,
    }
    gain_map = {(row["case_id"], int(row["actual_nfe"])): row[
        "rgb_l1_closure_gain"] for row in rows}

    bds_rows = []
    split_summary = []
    for split_name in SPLITS:
        calibration, holdout = make_split(split_name, case_ids)
        low_stats = None
        all_stats = None
        for tset_name, tset in tsets.items():
            stats = summarize_tset(gain_map, calibration, holdout, tset,
                                   n_boot)
            if tset_name == "BDS_low":
                low_stats = stats
            else:
                all_stats = stats
            bds_rows.append({
                "split": split_name,
                "tset": tset_name,
                "nfe_points": ",".join(str(x) for x in tset),
                "calibration_cases": len(stats["calibration_values"]),
                "holdout_cases": len(stats["holdout_values"]),
                "calibration_mean_gain": stats["calibration_mean"],
                "calibration_lcb_95": stats["lcb"],
                "calibration_ci_95_low": stats["ci_low"],
                "calibration_ci_95_high": stats["ci_high"],
                "calibration_win_rate": stats["calibration_win_rate"],
                "holdout_mean_gain": stats["holdout_mean"],
                "holdout_win_rate": stats["holdout_win_rate"],
                "notes": all_note if tset_name == "BDS_all" else "",
            })
        verdict = verdict_from_stats(low_stats, all_stats)
        split_summary.append({
            "split": split_name,
            "predicted_verdict": verdict,
            "low_lcb": low_stats["lcb"] if low_stats else float("nan"),
            "all_lcb": all_stats["lcb"] if all_stats else float("nan"),
            "low_mean": low_stats["calibration_mean"] if low_stats else float("nan"),
            "all_mean": all_stats["calibration_mean"] if all_stats else float("nan"),
            "holdout_low_mean": low_stats["holdout_mean"] if low_stats else float("nan"),
            "holdout_all_mean": all_stats["holdout_mean"] if all_stats else float("nan"),
            "low_confirmed": bool(low_stats and low_stats["holdout_mean"] > 0),
            "all_green_confirmed": bool(all_stats and all_stats["holdout_mean"] > 0),
            "low_only_confirmed": bool(low_stats and low_stats["holdout_mean"] > 0 and all_stats and all_stats["holdout_mean"] <= 0),
        })
    return bds_rows, split_summary


def summarize_tset(gain_map, calibration, holdout, tset, n_boot):
    calibration_values = case_values(gain_map, calibration, tset)
    holdout_values = case_values(gain_map, holdout, tset)
    boot = bootstrap_means(calibration_values, n_boot)
    return {
        "calibration_values": calibration_values,
        "holdout_values": holdout_values,
        "calibration_mean": mean_or_nan(calibration_values),
        "lcb": percentile_or_nan(boot, 2.5),
        "ci_low": percentile_or_nan(boot, 2.5),
        "ci_high": percentile_or_nan(boot, 97.5),
        "calibration_win_rate": win_rate(calibration_values),
        "holdout_mean": mean_or_nan(holdout_values),
        "holdout_win_rate": win_rate(holdout_values),
    }


def case_values(gain_map, cases, tset):
    values = []
    for case_id in cases:
        gains = [
            gain_map[(case_id, nfe)] for nfe in tset
            if (case_id, nfe) in gain_map
        ]
        if gains:
            values.append(mean(gains))
    return values


def bootstrap_means(values, n_boot):
    if not values:
        return []
    rng = random.Random(0)
    boot = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(sample))
    return boot


def make_split(name, case_ids):
    if name == "first_half_split":
        mid = len(case_ids) // 2
        return case_ids[:mid], case_ids[mid:]
    if name == "alternating_split":
        return case_ids[::2], case_ids[1::2]
    if name == "random_split_seed0":
        rng = random.Random(0)
        shuffled = list(case_ids)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        return sorted(shuffled[:mid]), sorted(shuffled[mid:])
    raise ValueError(name)


def verdict_from_stats(low_stats, all_stats):
    if not low_stats or len(low_stats["calibration_values"]) < 4:
        return "Need more data"
    if not all_stats or len(all_stats["calibration_values"]) < 4:
        if low_stats["lcb"] > 0:
            return "Low-only"
        return "Reject"
    if all_stats["lcb"] > 0:
        return "Green"
    if low_stats["lcb"] > 0:
        return "Low-only"
    return "Reject"


def final_verdict(split_summary):
    if not split_summary:
        return "Need more data"
    low_lcbs = [row["low_lcb"] for row in split_summary]
    all_lcbs = [row["all_lcb"] for row in split_summary]
    if any(math.isnan(x) for x in low_lcbs):
        return "Need more data"
    if all(not math.isnan(x) and x > 0 for x in all_lcbs):
        return "Green"
    if all(x > 0 for x in low_lcbs):
        return "Low-only"
    return "Reject"


def write_outputs(output_root: Path, bds_rows, split_summary):
    tables = output_root / "tables"
    reports = output_root / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "tableA_wan21_13b_bds_by_split.csv", bds_rows)
    write_markdown_table(tables / "tableA_wan21_13b_bds_by_split.md",
                         bds_rows)
    cross = build_cross_model_bds_row(bds_rows, split_summary)
    write_csv(tables / "cross_model_bds_row.csv", [cross])
    write_markdown_table(tables / "cross_model_bds_row.md", [cross])
    write_report(reports / "03_bds_report.md", bds_rows, split_summary,
                 cross)


def build_cross_model_bds_row(bds_rows, split_summary):
    low_rows = [row for row in bds_rows if row["tset"] == "BDS_low"]
    all_rows = [row for row in bds_rows if row["tset"] == "BDS_all"]
    tested = sorted({
        int(nfe)
        for row in all_rows
        for nfe in row["nfe_points"].split(",")
        if nfe
    })
    verdict = final_verdict(split_summary)
    return {
        "model_id": "Wan2.1",
        "model_variant": "T2V-1.3B",
        "setting": "Video Generation",
        "few_step_prior": "None / not explicit",
        "task": "t2v",
        "modality": "video",
        "protocol": "fixed_prompt_suite_official_script",
        "reference_method": "reference_uniform50",
        "reference_nfe": 50,
        "cases": max((int(row["calibration_cases"]) + int(row["holdout_cases"]) for row in low_rows), default=0),
        "tested_nfe_points": ",".join(str(x) for x in tested),
        "bds_low_mean_over_splits": mean_field(low_rows, "calibration_mean_gain"),
        "bds_low_lcb_min_over_splits": min_field(low_rows, "calibration_lcb_95"),
        "bds_all_mean_over_splits": mean_field(all_rows, "calibration_mean_gain"),
        "bds_all_lcb_min_over_splits": min_field(all_rows, "calibration_lcb_95"),
        "holdout_low_mean_gain_mean_over_splits": mean_field(low_rows, "holdout_mean_gain"),
        "holdout_all_mean_gain_mean_over_splits": mean_field(all_rows, "holdout_mean_gain"),
        "predicted_deployment": verdict,
        "final_verdict": verdict,
        "notes": "Fixed prompt suite using official Wan2.1 script; no universal claim.",
    }


def write_report(path: Path, bds_rows, split_summary, cross):
    lines = [
        "# Wan2.1-T2V-1.3B BDS Report",
        "",
        f"- Final verdict: {cross['final_verdict']}",
        f"- Tested NFE points: {cross['tested_nfe_points']}",
        f"- Cases: {cross['cases']}",
        "",
        "## Split Verdicts",
        "",
    ]
    for row in split_summary:
        lines.append(
            f"- {row['split']}: {row['predicted_verdict']} "
            f"(low LCB {format_value(row['low_lcb'])}, all LCB {format_value(row['all_lcb'])})"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_gain_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            row["actual_nfe"] = int(row["actual_nfe"])
            row["rgb_l1_closure_gain"] = float(row["rgb_l1_closure_gain"])
            rows.append(row)
        return rows


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(value) for key, value in row.items()})


def write_markdown_table(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(format_value(row.get(field, ""))) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_or_nan(values):
    return mean(values) if values else float("nan")


def percentile_or_nan(values, pct):
    return percentile(values, pct) if values else float("nan")


def win_rate(values):
    return mean([1 if value > 0 else 0 for value in values]) if values else float("nan")


def mean_field(rows, field):
    values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
    return mean(values) if values else float("nan")


def min_field(rows, field):
    values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
    return min(values) if values else float("nan")


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return value


def mean(values):
    return float(sum(values) / len(values))


def percentile(values, pct):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
