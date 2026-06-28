#!/usr/bin/env python3
"""Create final Wan2.1 same-compute tables, figures, and report."""

import argparse
import csv
import html
import math
import os
from pathlib import Path


NFE_LABELS = [
    (10, "Low"),
    (20, "Middle Low"),
    (30, "Middle High"),
    (40, "High"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_root = resolve_path(args.output_root, repo_dir)
    master = read_csv(output_root / "metrics/master_long_metrics.csv")
    gains = read_csv(output_root / "metrics/same_compute_gain_long.csv")
    bds = read_csv(output_root / "tables/cross_model_bds_row.csv")
    row, tex = build_same_compute_row(gains, bds)
    write_final_tables(output_root, row, tex)
    make_closure_plot(output_root, master)
    make_side_by_side(output_root, master, repo_dir)
    write_final_report(output_root, row, bds[0] if bds else None)
    print(f"Wrote final outputs under {output_root}")


def build_same_compute_row(gains, bds_rows):
    bds_verdict = bds_rows[0].get("final_verdict", "Need more data") if bds_rows else "Need more data"
    cases = len({row["case_id"] for row in gains})
    cells = {}
    deltas = []
    wins = []
    for nfe, label in NFE_LABELS:
        rows = [row for row in gains if int(row["actual_nfe"]) == nfe]
        if not rows:
            cells[label] = "--"
            continue
        bss_closure = mean_float(rows, "bss_rgb_l1_closure")
        delta = mean_float(rows, "rgb_l1_closure_gain")
        deltas.append(delta)
        wins.extend(float(row["win"]) for row in rows)
        cells[label] = f"{bss_closure:.3f} / {delta:+.3f} [{nfe} NFE]"
    mean_delta = f"{mean(deltas):+.3f}" if deltas else "--"
    win = f"{mean(wins):.3f}" if wins else "--"
    row = {
        "Model": "Wan2.1",
        "Setting": "Video Generation",
        "Few-step prior": "None / not explicit",
        "Ref.": "50",
        "Cases": str(cases),
        "BDS": bds_verdict,
        "Low": cells["Low"],
        "Middle Low": cells["Middle Low"],
        "Middle High": cells["Middle High"],
        "High": cells["High"],
        "Mean Δ": mean_delta,
        "Win": win,
    }
    tex = (
        "Wan2.1\n"
        "& Video Generation\n"
        "& None / not explicit\n"
        "& 50\n"
        f"& {row['Cases']}\n"
        f"& {row['BDS']}\n"
        f"& {row['Low']}\n"
        f"& {row['Middle Low']}\n"
        f"& {row['Middle High']}\n"
        f"& {row['High']}\n"
        f"& {row['Mean Δ']}\n"
        f"& {row['Win']}\n"
        "\\\\\n"
    )
    return row, tex


def write_final_tables(output_root: Path, row, tex):
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "table_cross_model_same_compute_wan21_row.csv", [row])
    write_markdown_table(tables / "table_cross_model_same_compute_wan21_row.md",
                         [row])
    (tables / "table_cross_model_same_compute_wan21_row.tex").write_text(
        tex, encoding="utf-8")


def make_closure_plot(output_root: Path, master):
    if not master:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_root / "reports/figures_skipped.md").write_text(
            f"Matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, label in [("uniform", "uniform"), ("bss", "BSS")]:
        points = []
        for nfe in sorted({int(row["actual_nfe"]) for row in master}):
            rows = [
                row for row in master
                if row["method_family"] == family and int(row["actual_nfe"]) == nfe
            ]
            if not rows:
                continue
            values = [float(row["rgb_l1_closure"]) for row in rows]
            center = mean(values)
            sem = sample_sem(values)
            points.append((nfe / 50.0, center, sem))
        if not points:
            continue
        xs, ys, sems = zip(*points)
        ax.plot(xs, ys, marker="o", label=label)
        ax.fill_between(xs, [y - s for y, s in zip(ys, sems)],
                        [y + s for y, s in zip(ys, sems)], alpha=0.2)
    ax.set_xlabel("NFE / reference NFE")
    ax.set_ylabel("RGB-L1 closure")
    ax.set_title("Wan2.1-T2V-1.3B same-compute closure")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "compute_quality_rgb_closure.png", dpi=200)
    fig.savefig(fig_dir / "compute_quality_rgb_closure.pdf")
    plt.close(fig)


def make_side_by_side(output_root: Path, master, repo_dir: Path):
    if not master:
        return
    fig_dir = output_root / "figures/side_by_side"
    fig_dir.mkdir(parents=True, exist_ok=True)
    by_case = {}
    for row in master:
        by_case.setdefault(row["case_id"], {})[row["method"]] = row
    panels = [
        ["uniform8", "uniform10", "bss10", "reference_uniform50"],
        [
            "uniform10",
            "bss10",
            "uniform20",
            "bss20",
            "uniform30",
            "bss30",
            "uniform40",
            "bss40",
            "reference_uniform50",
        ],
    ]
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        "<title>Wan2.1 BSS side-by-side</title>",
        "<style>body{font-family:sans-serif;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{border:1px solid #ddd;padding:8px}video{width:100%}</style>",
        "<h1>Wan2.1-T2V-1.3B side-by-side</h1>",
    ]
    for case_id, methods in sorted(by_case.items()):
        lines.append(f"<h2>{html.escape(case_id)}</h2>")
        for panel in panels:
            available = [method for method in panel if method in methods]
            if not available:
                continue
            lines.append("<div class=\"grid\">")
            for method in available:
                path = resolve_path(methods[method]["output_path"], repo_dir)
                rel = os.path.relpath(path, fig_dir)
                lines.append("<div class=\"card\">")
                lines.append(f"<h3>{html.escape(method)}</h3>")
                lines.append(
                    f"<video controls muted src=\"{html.escape(str(rel))}\"></video>"
                )
                lines.append("</div>")
            lines.append("</div>")
    (fig_dir / "index.html").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


def write_final_report(output_root: Path, row, bds_row):
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    verdict = row["BDS"]
    include = "yes" if row["Cases"] != "0" and verdict != "Need more data" else "not yet"
    lines = [
        "# Final Wan2.1-T2V-1.3B BSS/BDS Report",
        "",
        "## Purpose",
        "",
        "Produce a same-compute RGB-L1 closure row for Wan2.1-T2V-1.3B using the official Wan2.1 script with a training-free BSS schedule.",
        "",
        "## Same-Compute Row",
        "",
        markdown_table([row]),
        "",
        "## BDS",
        "",
        f"- Final verdict: {verdict}",
        f"- Can include in paper table: {include}",
        "",
        "## Caveats",
        "",
        "- Fixed prompt suite, not an official benchmark.",
        "- Reference is uniform50, not ground truth.",
        "- NFE is used as the compute proxy.",
        "- Results depend on sample_solver and sample_shift.",
        "- No universal claim is made.",
        "",
        "## Paths",
        "",
        "- Paper row: `tables/table_cross_model_same_compute_wan21_row.md`",
        "- LaTeX row: `tables/table_cross_model_same_compute_wan21_row.tex`",
        "- BDS table: `tables/tableA_wan21_13b_bds_by_split.md`",
        "- Figure: `figures/compute_quality_rgb_closure.png`",
        "- Side-by-side: `figures/side_by_side/index.html`",
        "",
    ]
    if bds_row:
        lines.extend([
            "## BDS Cross-Model Row",
            "",
            markdown_table([bds_row]),
            "",
        ])
    (reports / "FINAL_WAN21_13B_BSS_BDS_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows):
    path.write_text(markdown_table(rows) + "\n", encoding="utf-8")


def markdown_table(rows):
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def mean_float(rows, field):
    return mean([float(row[field]) for row in rows])


def mean(values):
    return float(sum(values) / len(values))


def sample_sem(values):
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    variance = sum((value - center)**2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
