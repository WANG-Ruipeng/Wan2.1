# Wan2.1-T2V-1.3B Repo, Model, and Hardware Audit

- Timestamp UTC: 2026-06-28T09:34:36.832542+00:00
- Repo path: `/Users/warpwang/Documents/GitHub/Wan2.1`
- Output root: `/Users/warpwang/Documents/GitHub/Wan2.1/bss_experiments/wan21_13b_bss_bds_v1`
- Git remote: `origin	https://github.com/WANG-Ruipeng/Wan2.1.git (fetch) | origin	https://github.com/WANG-Ruipeng/Wan2.1.git (push) | upstream	https://github.com/Wan-Video/Wan2.1.git (fetch) | upstream	https://github.com/Wan-Video/Wan2.1.git (push)`
- Current branch: `main`
- Commit hash: `9737cba9c1c3c4d04b33fcad41c111989865d315`
- Dirty status: `dirty`
- Python executable: `/Library/Developer/CommandLineTools/usr/bin/python3`
- Python version: `3.9.6`
- Platform: `macOS-26.5.1-arm64-arm-64bit`
- PyTorch version: `unavailable: No module named 'torch'`
- CUDA version: `unavailable`
- CUDA available: `unavailable`
- GPU name: `unavailable`
- GPU VRAM: `unavailable`
- Pro6000-class GPU: `unknown`
- Checkpoint dir: `/Users/warpwang/Documents/GitHub/Wan2.1/Wan2.1-T2V-1.3B`
- Checkpoint exists: `False`
- Missing checkpoint files: `diffusion_pytorch_model.safetensors, models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth`
- `generate.py` exists: `True`
- Official example prompt exists: `True`
- Official smoke prompt: `Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.`
- `python generate.py --help` works: `False`

## CLI Argument Audit

- `--task`: `yes`
- `--ckpt_dir`: `yes`
- `--prompt`: `yes`
- `--sample_steps`: `yes`
- `--sample_shift`: `yes`
- `--sample_solver`: `yes`
- `--base_seed`: `yes`
- `--sample_guide_scale`: `yes`
- `--size`: `yes`
- `--frame_num`: `yes`
- `--save_file`: `yes`
- `--offload_model`: `yes`
- `--t5_cpu`: `yes`
- `--sampler_mode`: `yes`
- `--base_sample_steps`: `yes`
- `--split_pairs`: `yes`
- `--dump_schedule_json`: `yes`

## Defaults and Protocol

- sample_steps: 50 for T2V in generate.py _validate_args
- sample_shift: code default 5.0; README recommends 8-12 for T2V-1.3B
- sample_solver: unipc
- sample_guide_scale: code default 5.0; README recommends 6
- frame_num: 81 for T2V
- size: CLI default 1280*720; T2V-1.3B protocol uses 832*480
- T2V default sample_steps is 50: yes, from generate.py _validate_args.
- Wan2.1-T2V-1.3B few-step prior: None / not explicit in local README audit.
- Reference method: reference_uniform50.
- Main experiment resolution: 832*480.
- Main experiment guide scale: README-recommended 6.
- Main experiment sample shift: README-recommended 8 within the 8-12 range.

## Scheduler and NFE Audit

- Timesteps are built in `wan/text2video.py` with `FlowUniPCMultistepScheduler` or `FlowDPMSolverMultistepScheduler`.
- UniPC uses `set_timesteps(sampling_steps, shift=sample_shift)`.
- DPM++ uses `get_sampling_sigmas(sampling_steps, sample_shift)` then `retrieve_timesteps(..., sigmas=...)`.
- `sample_steps` determines the number of denoising loop iterations.
- This experiment counts one denoising loop iteration as one NFE, while CFG performs conditional and unconditional forwards per step.

Line references:

- `wan/text2video.py:26` contains `FlowUniPCMultistepScheduler`
- `wan/text2video.py:22` contains `FlowDPMSolverMultistepScheduler`
- `wan/text2video.py:274` contains `for _, t in enumerate(tqdm(timesteps))`
- `wan/text2video.py:289` contains `sample_scheduler.step`

## nvidia-smi

```
[Errno 2] No such file or directory: 'nvidia-smi'
```
## Checkpoint Download Instructions

The Wan2.1-T2V-1.3B checkpoint is missing or incomplete at:

`/Users/warpwang/Documents/GitHub/Wan2.1/Wan2.1-T2V-1.3B`

For Colab with Drive mounted, cache the large checkpoint on Drive once:

```bash
pip install -q "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /content/drive/MyDrive/Colab_Projects/Wan2.1-BSS-BDS/models/Wan-AI/Wan2.1-T2V-1.3B
```

Then copy or rsync it to local Colab disk before inference:

```bash
rsync -a /content/drive/MyDrive/Colab_Projects/Wan2.1-BSS-BDS/models/Wan-AI/Wan2.1-T2V-1.3B/ /content/models/Wan-AI/Wan2.1-T2V-1.3B/
```

## README Evidence Snippet Locations

- Model table and download commands: README.md around the Model Download section.
- T2V-1.3B command: README.md uses `--size 832*480`, `--sample_shift 8`, and `--sample_guide_scale 6`.
- README mentions T2V-1.3B: `True`

