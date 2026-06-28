#!/usr/bin/env python3
"""Run Wan2.1-T2V-1.3B manifest rows with resume-friendly logging."""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--oom_retry_memory_saving", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    manifest_path = resolve_path(args.manifest, repo_dir)
    ckpt_dir = resolve_path(args.ckpt_dir, repo_dir)
    rows, fieldnames = read_manifest(manifest_path)

    executed = 0
    for idx, row in enumerate(rows):
        if args.limit and executed >= args.limit:
            break
        output_path = resolve_path(row["output_path"], repo_dir)
        runtime_path = resolve_path(row["runtime_json_path"], repo_dir)
        if should_skip(row, output_path, runtime_path, args.force):
            continue
        print(f"\n=== Running {idx + 1}/{len(rows)}: {row['run_id']} ===")
        row["status"] = "running"
        row["error_message"] = ""
        write_manifest(manifest_path, rows, fieldnames)
        rc, error = run_row(row, repo_dir, ckpt_dir, args, memory_saving=False)
        if (rc != 0 and args.oom_retry_memory_saving and
                not (args.offload_model and args.t5_cpu)):
            if looks_like_oom(error):
                print("OOM-like failure detected; retrying with memory-saving flags.")
                rc, error = run_row(
                    row, repo_dir, ckpt_dir, args, memory_saving=True)
        row["status"] = "done" if rc == 0 and output_path.exists() else "failed"
        row["error_message"] = "" if row["status"] == "done" else error[:1000]
        write_manifest(manifest_path, rows, fieldnames)
        executed += 1
        if row["status"] != "done":
            print_failure_debug(row, repo_dir)
            raise RuntimeError(f"Row failed: {row['run_id']}: {error[:1000]}")

    print(f"Executed {executed} row(s). Manifest: {manifest_path}")


def run_row(row, repo_dir: Path, ckpt_dir: Path, args, memory_saving: bool):
    output_path = resolve_path(row["output_path"], repo_dir)
    schedule_path = resolve_path(row["schedule_json_path"], repo_dir)
    stdout_path = resolve_path(row["stdout_log_path"], repo_dir)
    stderr_path = resolve_path(row["stderr_log_path"], repo_dir)
    runtime_path = resolve_path(row["runtime_json_path"], repo_dir)
    for path in [output_path, schedule_path, stdout_path, stderr_path,
                 runtime_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(repo_dir / "generate.py"),
        "--task",
        row.get("wan_task") or "t2v-1.3B",
        "--size",
        row["size"],
        "--frame_num",
        str(row["frame_num"]),
        "--ckpt_dir",
        str(ckpt_dir),
        "--prompt",
        row["prompt"],
        "--sample_solver",
        row["sample_solver"],
        "--sample_steps",
        str(row["sample_steps"]),
        "--sample_shift",
        str(row["sample_shift"]),
        "--sample_guide_scale",
        str(row["sample_guide_scale"]),
        "--base_seed",
        str(row["base_seed"]),
        "--save_file",
        str(output_path),
        "--sampler_mode",
        row["sampler_mode"],
        "--dump_schedule_json",
        str(schedule_path),
    ]
    if row["sampler_mode"] == "bss":
        cmd.extend(["--base_sample_steps", str(row["base_sample_steps"])])
        cmd.extend(["--split_pairs", row["split_pairs"]])
    if args.offload_model or memory_saving:
        cmd.extend(["--offload_model", "True"])
    if args.t5_cpu or memory_saving:
        cmd.append("--t5_cpu")

    start = time.time()
    runtime = {
        "run_id": row["run_id"],
        "cmd": cmd,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "memory_saving_retry": memory_saving,
    }
    stdout_error = []
    with stdout_path.open("w", encoding="utf-8") as stdout_handle:
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            threads = [
                threading.Thread(
                    target=tee_stream,
                    args=(proc.stdout, stdout_handle, sys.stdout,
                          stdout_error),
                ),
                threading.Thread(
                    target=tee_stream,
                    args=(proc.stderr, stderr_handle, sys.stderr,
                          stdout_error),
                ),
            ]
            for thread in threads:
                thread.start()
            rc = proc.wait()
            for thread in threads:
                thread.join()

    duration = time.time() - start
    runtime.update({
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": rc,
        "duration_sec": duration,
        "output_exists": output_path.exists(),
        "schedule_exists": schedule_path.exists(),
    })
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n",
                            encoding="utf-8")
    error_text = "".join(stdout_error[-200:])
    return rc, error_text


def tee_stream(source, log_handle, console, errors):
    if source is None:
        return
    for line in source:
        log_handle.write(line)
        log_handle.flush()
        console.write(line)
        console.flush()
        errors.append(line)


def should_skip(row, output_path: Path, runtime_path: Path, force: bool):
    if force:
        return False
    if row.get("status") == "done" and output_path.exists():
        print(f"Skipping done row: {row['run_id']}")
        return True
    if output_path.exists() and runtime_path.exists():
        print(f"Skipping existing output: {row['run_id']}")
        row["status"] = "done"
        return True
    return False


def looks_like_oom(text: str):
    lowered = text.lower()
    return any(token in lowered for token in [
        "out of memory",
        "cuda error",
        "cublas",
        "allocate",
        "memory",
    ])


def print_failure_debug(row, repo_dir: Path):
    print("\n===== Wan2.1 BSS/BDS row failure debug =====")
    print(f"run_id: {row.get('run_id')}")
    print(f"method: {row.get('method')}")
    print(f"case_id: {row.get('case_id')}")
    for label, field in [
        ("stdout", "stdout_log_path"),
        ("stderr", "stderr_log_path"),
        ("runtime", "runtime_json_path"),
        ("output", "output_path"),
        ("schedule", "schedule_json_path"),
    ]:
        path = resolve_path(row[field], repo_dir)
        print(f"{label}: {path}  exists={path.exists()}")
        if label in {"stdout", "stderr", "runtime"} and path.exists():
            print(f"\n----- tail {label} -----")
            print(tail_text(path, 8000))
    print("===== end failure debug =====\n")


def tail_text(path: Path, max_chars: int):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"
    return text[-max_chars:]


def read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, reader.fieldnames or []


def write_manifest(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_text, repo_dir: Path):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_dir / path


if __name__ == "__main__":
    main()
