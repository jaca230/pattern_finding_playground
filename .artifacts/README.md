# Artifact Notes

The notebooks in this directory were originally written around the older grouped-hit
prototype parquet schema (`hits_time_group`, `hits_strip_type`, `positron_px`, and
similar columns).

The current DetResponse ML parquet uses the newer PURITY-style schema instead:

- ATAR hit columns: `atar_x`, `atar_y`, `atar_z`, `atar_t`, `atar_E`, `atar_view`
- LYSO hit columns: `lyso_x`, `lyso_y`, `lyso_z`, `lyso_t`, `lyso_E`
- Event truth: `truth_theta`, `truth_phi`, `truth_positron_energy`, and the
  `truth_*_start_*` / `truth_*_stop_*` coordinates
- Time slicing: `atar_slice_id`, `atar_slice_mean_t`, `lyso_slice`, `lyso_slice_mean_t`

For current parquet debugging, prefer the Python script:

```bash
python3 /workdir/playground/reco_algorithm_tests/.artifacts/inspect_purity_parquet_alignment.py \
    /workdir/all_ml_000.parquet --row 42
```

That script is designed around the current schema and prints slice-level diagnostics
before plotting XZ/YZ overlays using the event-level positron truth direction derived
from `truth_theta` and `truth_phi`.
