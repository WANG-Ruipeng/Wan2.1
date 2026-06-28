#!/usr/bin/env python3
"""Compute Wan2.1 video metrics against uniform50 references."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    manifest_path = resolve_path(args.manifest, repo_dir)
    output_root = resolve_path(args.output_root, repo_dir)
    rows = read_manifest(manifest_path)
    metrics = compute_all(rows, repo_dir)
    output_metrics(output_root, metrics)
    print(f"Wrote metrics under {output_root / 'metrics'}")


def compute_all(rows, repo_dir: Path):
    done_rows = [
        row for row in rows
        if row.get("status") == "done"
        and resolve_path(row["output_path"], repo_dir).exists()
    ]
    by_case = {}
    for row in done_rows:
        by_case.setdefault(row["case_id"], {})[row["method"]] = row

    metrics = []
    for case_id, methods in by_case.items():
        ref_row = methods.get("reference_uniform50")
        base_row = methods.get("uniform8")
        if not ref_row or not base_row:
            continue
        ref = read_video(resolve_path(ref_row["output_path"], repo_dir))
        base = read_video(resolve_path(base_row["output_path"], repo_dir))
        base_aligned, ref_for_base = align_arrays(base, ref)
        base_l1 = rgb_l1(base_aligned, ref_for_base)
        base_temporal = temporal_ref_error(base_aligned, ref_for_base)
        for method, row in methods.items():
            video = ref if method == "reference_uniform50" else read_video(
                resolve_path(row["output_path"], repo_dir))
            arr, ref_arr = align_arrays(video, ref)
            l1 = rgb_l1(arr, ref_arr)
            l2 = rgb_l2(arr, ref_arr)
            mse = l2
            temporal_l1 = temporal_diff_l1(arr)
            temporal_ref = temporal_ref_error(arr, ref_arr)
            metrics.append({
                "case_id": case_id,
                "method": method,
                "method_family": row["method_family"],
                "actual_nfe": int(row["actual_nfe"]),
                "compute_fraction": int(row["actual_nfe"]) / 50.0,
                "rgb_l1_to_ref": l1,
                "rgb_l2_to_ref": l2,
                "psnr_to_ref": psnr(mse),
                "temporal_diff_l1": temporal_l1,
                "temporal_diff_l1_to_ref": temporal_ref,
                "rgb_l1_closure": closure(l1, base_l1),
                "temporal_closure": closure(temporal_ref, base_temporal),
                "runtime_sec": read_runtime(row, repo_dir),
                "output_path": row["output_path"],
                "reference_method": "reference_uniform50",
                "low_baseline_method": "uniform8",
                "prompt_hash": row["prompt_hash"],
            })
    return metrics


def output_metrics(output_root: Path, metrics):
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "master_long_metrics.csv", metrics)
    write_csv(metrics_dir / "per_case_metrics.csv", metrics)
    gain_rows = same_compute_gain(metrics)
    write_csv(metrics_dir / "same_compute_gain_long.csv", gain_rows)


def same_compute_gain(metrics):
    by_case_nfe = {}
    for row in metrics:
        by_case_nfe.setdefault((row["case_id"], row["actual_nfe"]), {})[
            row["method_family"]] = row
    gains = []
    for (case_id, nfe), families in sorted(by_case_nfe.items()):
        bss = families.get("bss")
        uniform = families.get("uniform")
        if not bss or not uniform:
            continue
        gain = bss["rgb_l1_closure"] - uniform["rgb_l1_closure"]
        gains.append({
            "case_id": case_id,
            "actual_nfe": nfe,
            "bss_method": bss["method"],
            "uniform_method": uniform["method"],
            "bss_rgb_l1_closure": bss["rgb_l1_closure"],
            "uniform_rgb_l1_closure": uniform["rgb_l1_closure"],
            "rgb_l1_closure_gain": gain,
            "win": 1 if gain > 0 else 0,
            "bss_runtime_sec": bss["runtime_sec"],
            "uniform_runtime_sec": uniform["runtime_sec"],
        })
    return gains


def read_video(path: Path):
    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
    except Exception:
        import imageio

        arr = np.asarray(imageio.mimread(path))
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.dtype.kind in "ui":
        max_value = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / float(max_value)
    else:
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 2.0:
            arr = arr / 255.0
    return arr


def align_arrays(a, b):
    frames = min(a.shape[0], b.shape[0])
    height = min(a.shape[1], b.shape[1])
    width = min(a.shape[2], b.shape[2])
    channels = min(a.shape[3], b.shape[3])
    return (a[:frames, :height, :width, :channels],
            b[:frames, :height, :width, :channels])


def rgb_l1(a, b):
    return float(np.mean(np.abs(a - b)))


def rgb_l2(a, b):
    return float(np.mean((a - b)**2))


def psnr(mse):
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def temporal_diff_l1(a):
    if a.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(a, axis=0))))


def temporal_ref_error(a, ref):
    if a.shape[0] < 2 or ref.shape[0] < 2:
        return float("nan")
    da = np.diff(a, axis=0)
    dr = np.diff(ref, axis=0)
    frames = min(da.shape[0], dr.shape[0])
    return float(np.mean(np.abs(da[:frames] - dr[:frames])))


def closure(distance, baseline_distance):
    if baseline_distance is None or abs(baseline_distance) < 1e-12:
        return float("nan")
    return float(1.0 - distance / baseline_distance)


def read_runtime(row, repo_dir: Path):
    path = resolve_path(row["runtime_json_path"], repo_dir)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("duration_sec", "")
    except Exception:
        return ""


def read_manifest(path: Path):
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
        for row in rows:
            writer.writerow(format_row(row))


def format_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isinf(value):
                out[key] = "inf"
            elif math.isnan(value):
                out[key] = "nan"
            else:
                out[key] = f"{value:.10g}"
        else:
            out[key] = value
    return out


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
