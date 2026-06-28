#!/usr/bin/env python3
"""Validate Wan2.1 BSS schedule JSON files."""

import argparse
import csv
import json
from pathlib import Path


EXPECTED_METHOD_NFE = {
    "uniform8": 8,
    "uniform10": 10,
    "uniform20": 20,
    "uniform30": 30,
    "uniform40": 40,
    "reference_uniform50": 50,
    "bss10": 10,
    "bss20": 20,
    "bss30": 30,
    "bss40": 40,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    parser.add_argument("--report_name", default="01_wan21_13b_smoke_report.md")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    manifest_path = resolve_path(args.manifest, repo_dir)
    output_root = resolve_path(args.output_root, repo_dir)
    rows = read_manifest(manifest_path)
    results, schedules = validate_rows(rows, repo_dir)
    pair_results = validate_same_nfe_pairs(schedules)
    summary_path = output_root / "metrics/schedule_validation_summary.csv"
    write_summary(summary_path, results + pair_results)
    report_path = output_root / f"reports/{args.report_name}"
    write_report(report_path, manifest_path, results, pair_results)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def validate_rows(rows, repo_dir: Path):
    results = []
    schedules = {}
    for row in rows:
        method = row["method"]
        case_id = row["case_id"]
        schedule_path = resolve_path(row["schedule_json_path"], repo_dir)
        status = "pass"
        notes = []
        payload = None
        if not schedule_path.exists():
            status = "missing"
            notes.append("schedule_json_missing")
        else:
            payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            expected_nfe = EXPECTED_METHOD_NFE.get(method, int(row["actual_nfe"]))
            final_coords = payload.get("final_coords", [])
            actual_nfe = payload.get("actual_nfe")
            if actual_nfe != expected_nfe:
                status = "fail"
                notes.append(f"actual_nfe {actual_nfe} != {expected_nfe}")
            if len(final_coords) != expected_nfe:
                status = "fail"
                notes.append(
                    f"final_coords len {len(final_coords)} != {expected_nfe}")
            if method.startswith("bss"):
                ok, msg = validate_bss_payload(payload)
                if not ok:
                    status = "fail"
                    notes.append(msg)
            if row["prompt_hash"] != payload.get("prompt_hash"):
                status = "fail"
                notes.append("prompt_hash_mismatch")
            if str(row["base_seed"]) != str(payload.get("base_seed")):
                status = "fail"
                notes.append("base_seed_mismatch")
            schedules[(case_id, method)] = payload
        results.append({
            "case_id": case_id,
            "method": method,
            "check": "schedule_json",
            "status": status,
            "notes": "; ".join(notes),
            "path": str(schedule_path),
        })
    return results, schedules


def validate_bss_payload(payload):
    base_coords = payload.get("base_coords", [])
    final_coords = payload.get("final_coords", [])
    inserted = payload.get("inserted_midpoints", [])
    base_steps = payload.get("base_sample_steps")
    if base_steps != len(base_coords):
        return False, "base_sample_steps does not match base_coords"
    if len(inserted) != 2:
        return False, "BSS should insert exactly two midpoints"
    if len(final_coords) != len(base_coords) + len(inserted):
        return False, "BSS final length is not base + inserted"
    for item in inserted:
        expected = 0.5 * (float(item["left_coord"]) +
                          float(item["right_coord"]))
        if abs(float(item["midpoint"]) - expected) > 1e-6:
            return False, "inserted midpoint is not interval average"
        pos = int(item["inserted_position"])
        if abs(float(final_coords[pos]) - float(item["midpoint"])) > 1e-6:
            return False, "midpoint position mismatch"
    interval_indices = [int(item["interval_index"]) for item in inserted]
    if interval_indices != [0, len(base_coords) - 1]:
        return False, "BSS did not split first and last intervals"
    return True, ""


def validate_same_nfe_pairs(schedules):
    rows = []
    for (case_id, method), payload in schedules.items():
        if not method.startswith("bss"):
            continue
        nfe = payload.get("actual_nfe")
        uniform = schedules.get((case_id, f"uniform{nfe}"))
        if uniform is None:
            continue
        same = payload.get("final_coords") == uniform.get("final_coords")
        rows.append({
            "case_id": case_id,
            "method": method,
            "check": "bss_not_uniform_same_nfe",
            "status": "fail" if same else "pass",
            "notes": "" if not same else "BSS schedule equals uniform schedule",
            "path": "",
        })
    return rows


def write_summary(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "method", "check", "status", "notes", "path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, manifest_path: Path, results, pair_results):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = results + pair_results
    pass_count = sum(1 for row in all_rows if row["status"] == "pass")
    fail_count = sum(1 for row in all_rows if row["status"] == "fail")
    missing_count = sum(1 for row in all_rows if row["status"] == "missing")
    lines = [
        "# Wan2.1-T2V-1.3B Schedule Validation Report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Checks passed: {pass_count}",
        f"- Checks failed: {fail_count}",
        f"- Checks missing: {missing_count}",
        "",
        "## Failures or Missing",
        "",
    ]
    problem_rows = [row for row in all_rows if row["status"] != "pass"]
    if not problem_rows:
        lines.append("No schedule validation failures.")
    else:
        for row in problem_rows:
            lines.append(
                f"- {row['case_id']} {row['method']} {row['check']}: "
                f"{row['status']} {row['notes']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
