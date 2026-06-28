#!/usr/bin/env python3
"""Prompt-level Wan2.1 BSS gain diagnostics.

This is a pure analysis pass over existing metrics and videos. It does not run
inference or touch model weights.
"""

import argparse
import csv
import html
import math
import os
from pathlib import Path


NFE_POINTS = [10, 20, 30, 40]

ATTRIBUTE_MAP = {
    "p001_simple_object_motion": {
        "camera_motion": 0,
        "subject_motion": 0,
        "object_motion": 1,
        "high_texture": 0,
        "lighting_heavy": 1,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 0,
    },
    "p002_human_motion": {
        "camera_motion": 0,
        "subject_motion": 1,
        "object_motion": 0,
        "high_texture": 0,
        "lighting_heavy": 0,
        "indoor": 0,
        "outdoor": 1,
        "temporal_challenge": 1,
    },
    "p003_camera_movement": {
        "camera_motion": 1,
        "subject_motion": 0,
        "object_motion": 0,
        "high_texture": 1,
        "lighting_heavy": 0,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 1,
    },
    "p004_lighting_heavy": {
        "camera_motion": 0,
        "subject_motion": 0,
        "object_motion": 1,
        "high_texture": 0,
        "lighting_heavy": 1,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 0,
    },
    "p005_outdoor_scene": {
        "camera_motion": 0,
        "subject_motion": 0,
        "object_motion": 1,
        "high_texture": 0,
        "lighting_heavy": 0,
        "indoor": 0,
        "outdoor": 1,
        "temporal_challenge": 0,
    },
    "p006_indoor_scene": {
        "camera_motion": 0,
        "subject_motion": 0,
        "object_motion": 1,
        "high_texture": 1,
        "lighting_heavy": 1,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 1,
    },
    "p007_high_frequency_detail": {
        "camera_motion": 0,
        "subject_motion": 1,
        "object_motion": 1,
        "high_texture": 1,
        "lighting_heavy": 0,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 1,
    },
    "p008_temporal_consistency": {
        "camera_motion": 0,
        "subject_motion": 0,
        "object_motion": 1,
        "high_texture": 0,
        "lighting_heavy": 1,
        "indoor": 1,
        "outdoor": 0,
        "temporal_challenge": 1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_root = resolve_path(args.output_root, repo_dir)
    gains = read_csv(output_root / "metrics/same_compute_gain_long.csv")
    master = read_csv(output_root / "metrics/master_long_metrics.csv")
    prompt_rows = read_prompt_rows(output_root)
    manifest_rows = read_manifest_rows(output_root)

    per_prompt = build_per_prompt_table(gains, prompt_rows)
    attr_summary = build_attribute_summary(per_prompt)

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    gallery_dir = figures_dir / "side_by_side"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    write_csv(tables_dir / "per_prompt_gain_table.csv", per_prompt)
    write_markdown_table(tables_dir / "per_prompt_gain_table.md", per_prompt)
    write_csv(tables_dir / "prompt_attribute_gain_summary.csv", attr_summary)
    write_markdown_table(tables_dir / "prompt_attribute_gain_summary.md",
                         attr_summary)
    make_waterfall_plot(figures_dir / "gain_waterfall_10_20_30_40.png",
                        per_prompt)
    write_failure_gallery(gallery_dir / "failure_prompt_gallery.html",
                          per_prompt, master, manifest_rows, repo_dir)
    write_report(reports_dir / "04_prompt_gain_diagnostics.md", per_prompt,
                 attr_summary)

    print(f"Wrote prompt diagnostics under {output_root}")


def build_per_prompt_table(gains, prompt_rows):
    by_case = {}
    for row in gains:
        try:
            nfe = int(row["actual_nfe"])
        except (TypeError, ValueError):
            continue
        if nfe not in NFE_POINTS:
            continue
        by_case.setdefault(row["case_id"], {})[nfe] = safe_float(
            row["rgb_l1_closure_gain"])

    rows = []
    for case_id in sorted(prompt_rows):
        prompt_info = prompt_rows[case_id]
        gains_by_nfe = by_case.get(case_id, {})
        attrs = ATTRIBUTE_MAP.get(case_id, {})
        row = {
            "case_id": case_id,
            "prompt": prompt_info.get("prompt", ""),
            "category": prompt_info.get("category", ""),
            "delta_at_10": format_float(gains_by_nfe.get(10)),
            "delta_at_20": format_float(gains_by_nfe.get(20)),
            "delta_at_30": format_float(gains_by_nfe.get(30)),
            "delta_at_40": format_float(gains_by_nfe.get(40)),
            "pattern": classify_pattern(gains_by_nfe),
            "mean_delta_available": format_float(mean(
                list(gains_by_nfe.values())) if gains_by_nfe else None),
            "worst_delta_available": format_float(min(
                gains_by_nfe.values()) if gains_by_nfe else None),
        }
        for attr, value in attrs.items():
            row[attr] = value
        rows.append(row)
    return rows


def build_attribute_summary(per_prompt):
    attributes = sorted(next(iter(ATTRIBUTE_MAP.values())).keys())
    rows = []
    for attr in attributes:
        selected = [row for row in per_prompt if int(row.get(attr, 0)) == 1]
        for nfe in NFE_POINTS:
            key = f"delta_at_{nfe}"
            values = [safe_float(row[key]) for row in selected]
            values = [value for value in values if value is not None]
            rows.append({
                "attribute": attr,
                "nfe": nfe,
                "num_prompts_with_attribute": len(selected),
                "num_available": len(values),
                "mean_delta": format_float(mean(values) if values else None),
                "win_rate": format_float(
                    mean([1.0 if value > 0 else 0.0
                          for value in values]) if values else None),
                "min_delta": format_float(min(values) if values else None),
                "max_delta": format_float(max(values) if values else None),
            })
    return rows


def classify_pattern(gains_by_nfe):
    values = [gains_by_nfe.get(nfe) for nfe in NFE_POINTS]
    signs = [sign(value) for value in values]
    available = [value for value in values if value is not None]
    if not available:
        return "missing"
    if all(value > 0 for value in available):
        return "all-positive"
    if all(value <= 0 for value in available):
        return "all-negative"
    low_positive = all((gains_by_nfe.get(nfe, -1) > 0) for nfe in [10, 20]
                       if nfe in gains_by_nfe)
    high_negative = all((gains_by_nfe.get(nfe, 1) <= 0) for nfe in [30, 40]
                        if nfe in gains_by_nfe)
    if low_positive and high_negative:
        return "low-only"
    if gains_by_nfe.get(40) is not None and gains_by_nfe[40] <= 0:
        if any(gains_by_nfe.get(nfe, -1) > 0 for nfe in [10, 20, 30]):
            return "high-negative"
    if signs.count("+") >= 2 and signs[-1] == "-":
        return "mid-positive"
    return "unstable"


def make_waterfall_plot(path: Path, per_prompt):
    rows = [row for row in per_prompt if row["pattern"] != "missing"]
    if not rows:
        path.write_text("No gain data available for plot.\n", encoding="utf-8")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".txt").write_text(
            f"Matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    fig, axes = plt.subplots(
        len(NFE_POINTS), 1, figsize=(10, 2.2 * len(NFE_POINTS)), sharex=True)
    if len(NFE_POINTS) == 1:
        axes = [axes]
    labels = [row["case_id"].replace("p00", "p") for row in rows]
    for ax, nfe in zip(axes, NFE_POINTS):
        key = f"delta_at_{nfe}"
        values = [safe_float(row[key]) for row in rows]
        colors = ["#2f8f5b" if value and value > 0 else "#b84a4a"
                  for value in values]
        ax.bar(labels, [value if value is not None else 0 for value in values],
               color=colors)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_ylabel(f"Delta@{nfe}")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Prompt")
    fig.suptitle("Wan2.1-T2V-1.3B BSS same-compute gain waterfall")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_failure_gallery(path: Path, per_prompt, master, manifest_rows,
                          repo_dir: Path):
    by_case_method = {}
    for row in master:
        by_case_method.setdefault(row["case_id"], {})[row["method"]] = row
    for row in manifest_rows:
        by_case_method.setdefault(row["case_id"], {}).setdefault(
            row["method"], row)

    ranked = sorted(
        [row for row in per_prompt if row["pattern"] != "missing"],
        key=lambda row: safe_float(row["worst_delta_available"]) or 0,
    )
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        "<title>Wan2.1 prompt gain diagnostics</title>",
        "<style>body{font-family:sans-serif;margin:24px;line-height:1.4}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{border:1px solid #ddd;padding:8px}video{width:100%;background:#111}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 6px}</style>",
        "<h1>Wan2.1 prompt gain diagnostics</h1>",
        "<p>Worst prompts first by minimum same-compute BSS gain.</p>",
    ]
    for row in ranked:
        case_id = row["case_id"]
        lines.append(f"<h2>{html.escape(case_id)}: {html.escape(row['pattern'])}</h2>")
        lines.append(f"<p>{html.escape(row['prompt'])}</p>")
        lines.append("<table><tr><th>NFE</th><th>Delta</th></tr>")
        for nfe in NFE_POINTS:
            lines.append(
                f"<tr><td>{nfe}</td><td>{html.escape(row[f'delta_at_{nfe}'])}</td></tr>"
            )
        lines.append("</table>")
        lines.append("<div class=\"grid\">")
        for method in [
                "uniform10", "bss10", "uniform20", "bss20", "uniform30",
                "bss30", "uniform40", "bss40", "reference_uniform50"
        ]:
            item = by_case_method.get(case_id, {}).get(method)
            if not item:
                continue
            video_path = resolve_path(item.get("output_path", ""), repo_dir)
            rel = os.path.relpath(video_path, path.parent)
            lines.append("<div class=\"card\">")
            lines.append(f"<h3>{html.escape(method)}</h3>")
            lines.append(
                f"<video controls muted src=\"{html.escape(rel)}\"></video>")
            lines.append("</div>")
        lines.append("</div>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, per_prompt, attr_summary):
    patterns = {}
    for row in per_prompt:
        patterns[row["pattern"]] = patterns.get(row["pattern"], 0) + 1
    lines = [
        "# Prompt Gain Diagnostics",
        "",
        "Pure analysis over existing Wan2.1-T2V-1.3B same-compute metrics.",
        "",
        "## Pattern Counts",
        "",
    ]
    for pattern, count in sorted(patterns.items()):
        lines.append(f"- {pattern}: {count}")
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `tables/per_prompt_gain_table.csv`",
        "- `tables/prompt_attribute_gain_summary.csv`",
        "- `figures/gain_waterfall_10_20_30_40.png`",
        "- `figures/side_by_side/failure_prompt_gallery.html`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def read_prompt_rows(output_root: Path):
    path = output_root / "prompt_suites/wan21_13b_prompt_suite_v1.json"
    if path.exists():
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["case_id"]: item for item in payload.get("prompts", [])}
    return {}


def read_manifest_rows(output_root: Path):
    rows = []
    for name in [
            "wan21_13b_prompt_suite_manifest.csv",
            "wan21_13b_smoke_manifest.csv",
    ]:
        path = output_root / "manifests" / name
        rows.extend(read_csv(path))
    return rows


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sign(value):
    if value is None:
        return ""
    return "+" if value > 0 else "-"


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def format_float(value):
    if value is None:
        return ""
    return f"{value:+.6f}"


def mean(values):
    return sum(values) / len(values)


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
