# Air-to-Ground 500 m Campaign Configs

UAV-to-ground channel sounding at 3.4 GHz, **500 m link distance**, ZC-only waveforms.

## Variants

| File                       | Fs (MHz) | ZC SEQ_LEN | NUM_REPEATS | Range res | Unambig delay     | Proc. gain | Aperiodic PSLR |
|----------------------------|----------|-----------:|------------:|----------:|------------------:|-----------:|---------------:|
| `baseline_56mhz_*.yaml`    | 56       | 401        | 8           | 5.36 m    | 9.45 us / 2.83 km | 26.0 dB    | ~27 dB         |
| `highres_100mhz_*.yaml`    | 100      | 401        | 8           | 3.00 m    | 6.01 us / 1.80 km | 26.0 dB    | ~27 dB         |
| `longseq_56mhz_*.yaml`     | 56       | 1021       | 4           | 5.36 m    | 27.4 us / 8.21 km | 30.1 dB    | ~30 dB         |


## Verifying with the dry-run

From `sounder/`:

```bash
python3 dry_run_waveform.py -c ../config/a2g_500m/baseline_56mhz_tx.yaml \
    -o ../measurements/dryrun_baseline_56mhz.png
python3 dry_run_waveform.py -c ../config/a2g_500m/highres_100mhz_tx.yaml \
    -o ../measurements/dryrun_highres_100mhz.png
python3 dry_run_waveform.py -c ../config/a2g_500m/longseq_56mhz_tx.yaml \
    -o ../measurements/dryrun_longseq_56mhz.png
```

