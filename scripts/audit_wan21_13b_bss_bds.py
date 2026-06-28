#!/usr/bin/env python3
"""Write Wan2.1-1.3B repo/model/hardware audit reports."""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SUBDIRS = [
    "reports",
    "tables",
    "metrics",
    "figures",
    "schedules",
    "manifests",
    "logs",
    "scripts",
    "patches",
    "splits",
    "prompt_suites",
    "outputs/videos",
]

REQUIRED_CKPT_FILES = [
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
]

EXPECTED_ARGS = [
    "--task",
    "--ckpt_dir",
    "--prompt",
    "--sample_steps",
    "--sample_shift",
    "--sample_solver",
    "--base_seed",
    "--sample_guide_scale",
    "--size",
    "--frame_num",
    "--save_file",
    "--offload_model",
    "--t5_cpu",
    "--sampler_mode",
    "--base_sample_steps",
    "--split_pairs",
    "--dump_schedule_json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--output_root",
                        default="bss_experiments/wan21_13b_bss_bds_v1")
    parser.add_argument("--ckpt_dir", default="Wan2.1-T2V-1.3B")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_root = (repo_dir / args.output_root).resolve()
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_dir / ckpt_dir

    ensure_dirs(output_root)
    audit_path = output_root / "reports/00_repo_model_hardware_audit.md"
    wan22_path = output_root / "reports/00b_wan22_future_extension_audit.md"

    audit_path.write_text(build_main_audit(repo_dir, output_root, ckpt_dir),
                          encoding="utf-8")
    wan22_path.write_text(build_wan22_audit(repo_dir), encoding="utf-8")
    print(f"Wrote {audit_path}")
    print(f"Wrote {wan22_path}")


def ensure_dirs(output_root: Path):
    for subdir in SUBDIRS:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def build_main_audit(repo_dir: Path, output_root: Path, ckpt_dir: Path) -> str:
    remote = run_text(["git", "remote", "-v"], repo_dir)
    branch = run_text(["git", "branch", "--show-current"], repo_dir)
    commit = run_text(["git", "rev-parse", "HEAD"], repo_dir)
    dirty = run_text(["git", "status", "--porcelain"], repo_dir)
    nvidia = run_text(["nvidia-smi"], repo_dir, check=False)
    gpu_name, gpu_vram = parse_nvidia_smi(nvidia)
    torch_info = get_torch_info()
    help_text = run_text([sys.executable, "generate.py", "--help"],
                         repo_dir,
                         check=False)
    generate_py = repo_dir / "generate.py"
    text2video_py = repo_dir / "wan/text2video.py"
    readme = (repo_dir / "README.md").read_text(
        encoding="utf-8", errors="replace")

    arg_status = []
    for name in EXPECTED_ARGS:
        arg_status.append((name, has_arg(help_text, generate_py, name)))

    ckpt_exists = ckpt_dir.is_dir()
    missing_ckpt_files = [
        name for name in REQUIRED_CKPT_FILES if not (ckpt_dir / name).exists()
    ]

    defaults = {
        "sample_steps": "50 for T2V in generate.py _validate_args",
        "sample_shift":
            "code default 5.0; README recommends 8-12 for T2V-1.3B",
        "sample_solver": "unipc",
        "sample_guide_scale": "code default 5.0; README recommends 6",
        "frame_num": "81 for T2V",
        "size": "CLI default 1280*720; T2V-1.3B protocol uses 832*480",
    }

    official_prompt = extract_official_prompt(generate_py)
    scheduler_lines = find_line_refs(text2video_py, [
        "FlowUniPCMultistepScheduler",
        "FlowDPMSolverMultistepScheduler",
        "for _, t in enumerate(tqdm(timesteps))",
        "sample_scheduler.step",
    ])

    download_instructions = ""
    if not ckpt_exists or missing_ckpt_files:
        download_instructions = f"""

## Checkpoint Download Instructions

The Wan2.1-T2V-1.3B checkpoint is missing or incomplete at:

`{ckpt_dir}`

For Colab with Drive mounted, cache the large checkpoint on Drive once:

```bash
pip install -q "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /content/drive/MyDrive/Colab_Projects/Wan2.1-BSS-BDS/models/Wan-AI/Wan2.1-T2V-1.3B
```

Then copy or rsync it to local Colab disk before inference:

```bash
rsync -a /content/drive/MyDrive/Colab_Projects/Wan2.1-BSS-BDS/models/Wan-AI/Wan2.1-T2V-1.3B/ /content/models/Wan-AI/Wan2.1-T2V-1.3B/
```
"""

    lines = [
        "# Wan2.1-T2V-1.3B Repo, Model, and Hardware Audit",
        "",
        f"- Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Repo path: `{repo_dir}`",
        f"- Output root: `{output_root}`",
        f"- Git remote: `{one_line(remote)}`",
        f"- Current branch: `{branch or 'unknown'}`",
        f"- Commit hash: `{commit or 'unknown'}`",
        f"- Dirty status: `{'dirty' if dirty else 'clean'}`",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- PyTorch version: `{torch_info['torch_version']}`",
        f"- CUDA version: `{torch_info['cuda_version']}`",
        f"- CUDA available: `{torch_info['cuda_available']}`",
        f"- GPU name: `{gpu_name}`",
        f"- GPU VRAM: `{gpu_vram}`",
        f"- Pro6000-class GPU: `{is_pro6000(gpu_name)}`",
        f"- Checkpoint dir: `{ckpt_dir}`",
        f"- Checkpoint exists: `{ckpt_exists}`",
        f"- Missing checkpoint files: `{', '.join(missing_ckpt_files) if missing_ckpt_files else 'none'}`",
        f"- `generate.py` exists: `{generate_py.exists()}`",
        f"- Official example prompt exists: `{bool(official_prompt)}`",
        f"- Official smoke prompt: `{official_prompt}`",
        f"- `python generate.py --help` works: `{help_text.startswith('usage:')}`",
        "",
        "## CLI Argument Audit",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{'yes' if ok else 'no'}`" for name, ok in arg_status)
    lines.extend([
        "",
        "## Defaults and Protocol",
        "",
    ])
    lines.extend(f"- {key}: {value}" for key, value in defaults.items())
    lines.extend([
        "- T2V default sample_steps is 50: yes, from generate.py _validate_args.",
        "- Wan2.1-T2V-1.3B few-step prior: None / not explicit in local README audit.",
        "- Reference method: reference_uniform50.",
        "- Main experiment resolution: 832*480.",
        "- Main experiment guide scale: README-recommended 6.",
        "- Main experiment sample shift: README-recommended 8 within the 8-12 range.",
        "",
        "## Scheduler and NFE Audit",
        "",
        "- Timesteps are built in `wan/text2video.py` with `FlowUniPCMultistepScheduler` or `FlowDPMSolverMultistepScheduler`.",
        "- UniPC uses `set_timesteps(sampling_steps, shift=sample_shift)`.",
        "- DPM++ uses `get_sampling_sigmas(sampling_steps, sample_shift)` then `retrieve_timesteps(..., sigmas=...)`.",
        "- `sample_steps` determines the number of denoising loop iterations.",
        "- This experiment counts one denoising loop iteration as one NFE, while CFG performs conditional and unconditional forwards per step.",
        "",
        "Line references:",
        "",
    ])
    lines.extend(f"- {item}" for item in scheduler_lines)
    lines.extend([
        "",
        "## nvidia-smi",
        "",
        "```",
        nvidia.strip() or "nvidia-smi unavailable",
        "```",
    ])
    if download_instructions:
        lines.append(download_instructions.strip())
    lines.extend([
        "",
        "## README Evidence Snippet Locations",
        "",
        "- Model table and download commands: README.md around the Model Download section.",
        "- T2V-1.3B command: README.md uses `--size 832*480`, `--sample_shift 8`, and `--sample_guide_scale 6`.",
        f"- README mentions T2V-1.3B: `{bool(re.search('Wan2.1-T2V-1.3B', readme))}`",
        "",
    ])
    return "\n".join(lines) + "\n"


def build_wan22_audit(repo_dir: Path) -> str:
    candidates = [
        repo_dir.parent / "Wan2.2",
        Path("/content/Wan2.2"),
        Path("/content/drive/MyDrive/Colab_Projects/Wan2.2"),
        Path("/content/drive/MyDrive/Wan2.2"),
    ]
    ckpt_names = [
        "Wan2.2-T2V-A14B",
        "Wan2.2-TI2V-5B",
        "Wan2.2-I2V-A14B",
    ]
    lines = [
        "# Wan2.2 Future Extension Audit",
        "",
        f"- Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        "- Scope: audit only. Do not run Wan2.2 for this experiment.",
        "",
        "## Local Repo Candidates",
        "",
    ]
    for path in candidates:
        lines.append(f"- `{path}`: `{'exists' if path.exists() else 'missing'}`")
    lines.extend([
        "",
        "## Checkpoint Candidates",
        "",
    ])
    for name in ckpt_names:
        hits = []
        for root in candidates + [repo_dir.parent, Path("/content/models")]:
            candidate = root / name
            if candidate.exists():
                hits.append(str(candidate))
        lines.append(f"- {name}: `{', '.join(hits) if hits else 'missing'}`")
    lines.extend([
        "",
        "## Feasibility Note",
        "",
        "- Wan2.2 availability is a future extension only.",
        "- Feasibility depends on exact model size, checkpoint format, and available VRAM.",
        "- Complete Wan2.1-T2V-1.3B smoke and mini-suite first.",
        "",
    ])
    return "\n".join(lines)


def run_text(cmd, cwd: Path, check=True) -> str:
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return str(exc)


def get_torch_info():
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:
        return {
            "torch_version": f"unavailable: {exc}",
            "cuda_version": "unavailable",
            "cuda_available": "unavailable",
        }


def parse_nvidia_smi(text: str):
    if not text or "NVIDIA-SMI" not in text:
        return "unavailable", "unavailable"
    query = shutil.which("nvidia-smi")
    if query:
        proc = subprocess.run(
            [
                query,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first = proc.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in first.split(",")]
            if len(parts) >= 2:
                return parts[0], parts[1]
    return "detected, parse unavailable", "detected, parse unavailable"


def has_arg(help_text: str, generate_py: Path, name: str) -> bool:
    if name in help_text:
        return True
    try:
        return name in generate_py.read_text(encoding="utf-8")
    except Exception:
        return False


def extract_official_prompt(generate_py: Path) -> str:
    text = generate_py.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'"t2v-1\.3B":\s*\{\s*"prompt":\s*"([^"]+)"', text)
    return match.group(1) if match else ""


def find_line_refs(path: Path, needles):
    refs = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for needle in needles:
        for idx, line in enumerate(lines, start=1):
            if needle in line:
                refs.append(f"`{path.relative_to(path.parents[1])}:{idx}` contains `{needle}`")
                break
    return refs


def is_pro6000(gpu_name: str) -> str:
    if not gpu_name or gpu_name == "unavailable":
        return "unknown"
    text = gpu_name.lower().replace(" ", "")
    return "yes" if "pro6000" in text or "rtx6000" in text else "no"


def one_line(text: str) -> str:
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())


if __name__ == "__main__":
    main()
