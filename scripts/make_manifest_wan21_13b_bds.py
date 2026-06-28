#!/usr/bin/env python3
"""Create Wan2.1-T2V-1.3B BSS/BDS prompt suites and manifests."""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT = "wan21_13b_bss_bds_v1"
MODEL_ID = "Wan2.1"
MODEL_VARIANT = "T2V-1.3B"
WAN_TASK = "t2v-1.3B"
TASK = "t2v"
SETTING = "Video Generation"
PROTOCOL = "fixed_prompt_suite_official_script"
REFERENCE_METHOD = "reference_uniform50"
LOW_BASELINE_METHOD = "uniform8"
OFFICIAL_SMOKE_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)

SUBDIRS = [
    "reports",
    "tables",
    "metrics",
    "figures",
    "schedules",
    "manifests",
    "logs/stdout",
    "logs/stderr",
    "scripts",
    "patches",
    "splits",
    "prompt_suites",
    "outputs/videos",
    "runtime",
]

PROMPT_SUITE = [
    {
        "case_id": "p001_simple_object_motion",
        "category": "simple object motion",
        "prompt": "A red ceramic mug slowly slides across a polished wooden table, leaving a soft reflection as morning light enters the room.",
    },
    {
        "case_id": "p002_human_motion",
        "category": "animal/human motion",
        "prompt": "A dancer in a blue jacket spins once on a quiet city sidewalk while loose fabric and hair follow the motion naturally.",
    },
    {
        "case_id": "p003_camera_movement",
        "category": "camera movement",
        "prompt": "A smooth forward camera move travels through a narrow greenhouse aisle, passing rows of leafy plants and glass panels.",
    },
    {
        "case_id": "p004_lighting_heavy",
        "category": "lighting-heavy scene",
        "prompt": "A glass prism on black velvet splits a bright beam of light into colored bands that shimmer across the surface.",
    },
    {
        "case_id": "p005_outdoor_scene",
        "category": "outdoor scene",
        "prompt": "A small sailboat crosses a calm lake under a cloudy sky, with distant trees reflected in the water.",
    },
    {
        "case_id": "p006_indoor_scene",
        "category": "indoor scene",
        "prompt": "A cozy kitchen scene with steam rising from a kettle while sunlight moves across tiled walls and hanging utensils.",
    },
    {
        "case_id": "p007_high_frequency_detail",
        "category": "high-frequency texture/detail",
        "prompt": "A close-up of embroidered fabric with tiny metallic threads rippling gently as a hand lifts one corner.",
    },
    {
        "case_id": "p008_temporal_consistency",
        "category": "temporal consistency challenge",
        "prompt": "A row of identical white candles flickers in sequence while the camera stays locked and wax slowly melts.",
    },
]

FIELDNAMES = [
    "run_id",
    "model_id",
    "model_variant",
    "task",
    "wan_task",
    "modality",
    "setting",
    "protocol",
    "case_id",
    "prompt",
    "prompt_hash",
    "method",
    "method_family",
    "sampler_mode",
    "sample_solver",
    "actual_nfe",
    "sample_steps",
    "base_sample_steps",
    "sample_shift",
    "split_pairs",
    "sample_guide_scale",
    "base_seed",
    "size",
    "frame_num",
    "reference_method",
    "reference_nfe",
    "low_baseline_method",
    "output_path",
    "schedule_json_path",
    "stdout_log_path",
    "stderr_log_path",
    "runtime_json_path",
    "status",
    "error_message",
    "git_commit",
    "dirty_status",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--output_root",
                        default=f"bss_experiments/{EXPERIMENT}")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sample_shift", type=float, default=8.0)
    parser.add_argument("--sample_solver", default="unipc")
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    output_root = Path(args.output_root)
    ensure_dirs(output_root)
    write_prompt_suite(output_root)

    commit, dirty = git_state(repo_dir)
    smoke_rows = build_rows(
        cases=[{
            "case_id": "smoke_official_cats",
            "category": "official smoke",
            "prompt": OFFICIAL_SMOKE_PROMPT,
        }],
        methods=["uniform8", "uniform10", "bss10", REFERENCE_METHOD],
        args=args,
        output_root=output_root,
        commit=commit,
        dirty=dirty,
    )
    full_rows = build_rows(
        cases=PROMPT_SUITE,
        methods=[
            "uniform8",
            "uniform10",
            "uniform20",
            "uniform30",
            "uniform40",
            REFERENCE_METHOD,
            "bss10",
            "bss20",
            "bss30",
            "bss40",
        ],
        args=args,
        output_root=output_root,
        commit=commit,
        dirty=dirty,
    )
    write_csv(output_root / "manifests/wan21_13b_smoke_manifest.csv",
              smoke_rows)
    write_csv(output_root / "manifests/wan21_13b_prompt_suite_manifest.csv",
              full_rows)
    write_plan(output_root)
    print(f"Wrote {len(smoke_rows)} smoke rows")
    print(f"Wrote {len(full_rows)} mini-suite rows")


def ensure_dirs(output_root: Path):
    for subdir in SUBDIRS:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def write_prompt_suite(output_root: Path):
    payload = {
        "name": "wan21_13b_prompt_suite_v1",
        "description":
            "Fixed prompt suite using official Wan2.1 script; not an official demo suite.",
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "task": TASK,
        "base_seed": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompts": PROMPT_SUITE,
    }
    path = output_root / "prompt_suites/wan21_13b_prompt_suite_v1.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_rows(cases, methods, args, output_root: Path, commit: str,
               dirty: str):
    rows = []
    for case in cases:
        for method in methods:
            method_info = parse_method(method)
            run_id = f"{case['case_id']}_{method}"
            stem = safe_name(run_id)
            schedule_path = output_root / f"schedules/{stem}.json"
            row = {
                "run_id": run_id,
                "model_id": MODEL_ID,
                "model_variant": MODEL_VARIANT,
                "task": TASK,
                "wan_task": WAN_TASK,
                "modality": "video",
                "setting": SETTING,
                "protocol": PROTOCOL,
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "prompt_hash": prompt_hash(case["prompt"]),
                "method": method,
                "method_family": method_info["family"],
                "sampler_mode": method_info["sampler_mode"],
                "sample_solver": args.sample_solver,
                "actual_nfe": method_info["actual_nfe"],
                "sample_steps": method_info["actual_nfe"],
                "base_sample_steps": method_info["base_sample_steps"],
                "sample_shift": format_float(args.sample_shift),
                "split_pairs": method_info["split_pairs"],
                "sample_guide_scale": format_float(args.sample_guide_scale),
                "base_seed": args.base_seed,
                "size": args.size,
                "frame_num": args.frame_num,
                "reference_method": REFERENCE_METHOD,
                "reference_nfe": 50,
                "low_baseline_method": LOW_BASELINE_METHOD,
                "output_path": str(output_root / f"outputs/videos/{stem}.mp4"),
                "schedule_json_path": str(schedule_path),
                "stdout_log_path": str(output_root / f"logs/stdout/{stem}.log"),
                "stderr_log_path": str(output_root / f"logs/stderr/{stem}.log"),
                "runtime_json_path": str(output_root / f"runtime/{stem}.json"),
                "status": "pending",
                "error_message": "",
                "git_commit": commit,
                "dirty_status": dirty,
            }
            rows.append(row)
    return rows


def parse_method(method: str):
    if method == REFERENCE_METHOD:
        return {
            "family": "reference",
            "sampler_mode": "uniform",
            "actual_nfe": 50,
            "base_sample_steps": "",
            "split_pairs": "",
        }
    if method.startswith("uniform"):
        nfe = int(method.replace("uniform", ""))
        return {
            "family": "uniform",
            "sampler_mode": "uniform",
            "actual_nfe": nfe,
            "base_sample_steps": "",
            "split_pairs": "",
        }
    if method.startswith("bss"):
        nfe = int(method.replace("bss", ""))
        return {
            "family": "bss",
            "sampler_mode": "bss",
            "actual_nfe": nfe,
            "base_sample_steps": nfe - 2,
            "split_pairs": "0,-1",
        }
    raise ValueError(f"Unknown method {method}")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_plan(output_root: Path):
    path = output_root / "reports/PLAN.md"
    text = f"""# Wan2.1-T2V-1.3B BSS/BDS Plan

1. Run `scripts/audit_wan21_13b_bss_bds.py`.
2. Generate smoke and mini-suite manifests with `scripts/make_manifest_wan21_13b_bds.py`.
3. Run smoke rows first: uniform8, uniform10, bss10, reference_uniform50.
4. Validate schedule JSON and compute smoke metrics.
5. Run the 8-prompt mini-suite only after smoke passes.
6. Compute RGB-L1 closure, same-compute BSS gains, BDS splits, final table rows, and final report.

Experiment root: `{output_root}`
"""
    path.write_text(text, encoding="utf-8")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in "-_" else "_")
    return "".join(keep)


def format_float(value: float) -> str:
    return f"{value:g}"


def git_state(repo_dir: Path):
    commit = run_git(repo_dir, ["rev-parse", "HEAD"]) or "unknown"
    dirty_text = run_git(repo_dir, ["status", "--porcelain"])
    dirty = "dirty" if dirty_text else "clean"
    return commit, dirty


def run_git(repo_dir: Path, args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
