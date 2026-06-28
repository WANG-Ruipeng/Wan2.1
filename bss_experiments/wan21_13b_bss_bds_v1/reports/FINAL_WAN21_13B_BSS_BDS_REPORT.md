# Final Wan2.1-T2V-1.3B BSS/BDS Report

## Purpose

Produce a same-compute RGB-L1 closure row for Wan2.1-T2V-1.3B using the official Wan2.1 script with a training-free BSS schedule.

## Same-Compute Row

| Model | Setting | Few-step prior | Ref. | Cases | BDS | Low | Middle Low | Middle High | High | Mean Δ | Win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wan2.1 | Video Generation | None / not explicit | 50 | 0 | Need more data | -- | -- | -- | -- | -- | -- |

## BDS

- Final verdict: Need more data
- Can include in paper table: not yet

## Caveats

- Fixed prompt suite, not an official benchmark.
- Reference is uniform50, not ground truth.
- NFE is used as the compute proxy.
- Results depend on sample_solver and sample_shift.
- No universal claim is made.

## Paths

- Paper row: `tables/table_cross_model_same_compute_wan21_row.md`
- LaTeX row: `tables/table_cross_model_same_compute_wan21_row.tex`
- BDS table: `tables/tableA_wan21_13b_bds_by_split.md`
- Figure: `figures/compute_quality_rgb_closure.png`
- Side-by-side: `figures/side_by_side/index.html`

## BDS Cross-Model Row

| model_id | model_variant | setting | few_step_prior | task | modality | protocol | reference_method | reference_nfe | cases | tested_nfe_points | bds_low_mean_over_splits | bds_low_lcb_min_over_splits | bds_all_mean_over_splits | bds_all_lcb_min_over_splits | holdout_low_mean_gain_mean_over_splits | holdout_all_mean_gain_mean_over_splits | predicted_deployment | final_verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wan2.1 | T2V-1.3B | Video Generation | None / not explicit | t2v | video | fixed_prompt_suite_official_script | reference_uniform50 | 50 | 0 |  | nan | nan | nan | nan | nan | nan | Need more data | Need more data | Fixed prompt suite using official Wan2.1 script; no universal claim. |
