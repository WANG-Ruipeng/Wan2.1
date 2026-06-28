# Wan2.1-T2V-1.3B BSS/BDS Plan

1. Run `scripts/audit_wan21_13b_bss_bds.py`.
2. Generate smoke and mini-suite manifests with `scripts/make_manifest_wan21_13b_bds.py`.
3. Run smoke rows first: uniform8, uniform10, bss10, reference_uniform50.
4. Validate schedule JSON and compute smoke metrics.
5. Run the 8-prompt mini-suite only after smoke passes.
6. Compute RGB-L1 closure, same-compute BSS gains, BDS splits, final table rows, and final report.

Experiment root: `bss_experiments/wan21_13b_bss_bds_v1`
