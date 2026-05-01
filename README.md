# USRP Channel Sounder

A configurable real-time channel sounder for USRP devices, with TX/RX runtime in `sounder/` and analysis tools in `post_processing/`.

![USRP Channel Sounder preview](figures/preview.png)

## Repository Layout

- `sounder/`: runtime TX/RX code.
- `config/`: example configuration files.
- `figures/`: README preview and example output images.
- `post_processing/`: analysis scripts and notebooks.
- `requirements.txt`: runtime Python dependencies.
- `post_processing/requirements.txt`: analysis dependencies.

## Prerequisites

- Linux machine with USRP hardware.
- UHD driver 
- Python 3.10+ recommended.
- Build tools for Cython compilation (the RX module is `ChsRX.pyx`).

## Setup

### Option 1: Native install

```bash
sudo apt-get update -y
sudo apt-get install -y libuhd-dev uhd-host python3-dev build-essential
uhd_images_downloader

python3 -m pip install -r requirements.txt
```

### Option 2: Docker

```bash
sudo docker build --network=host -t ch-sounder .
sudo docker run -dit --network=host --privileged -v /dev:/dev --name sounder ch-sounder
sudo docker exec -it sounder bash
```

## Running the Sounder

Run from the `sounder/` directory:

```bash
cd sounder
python3 main.py -c ../config/tx_config.yaml
```

For RX mode:

```bash
cd sounder
python3 main.py -c ../config/rx_config.yaml
```

Optional flag:

```bash
python3 main.py -c ../config/rx_config.yaml --plot
```

## Dry Run (No USRP Required)

Generate a waveform preview plot (time domain, spectrogram, cross-correlation,
configuration readout) directly from a YAML config:

```bash
cd sounder
python3 dry_run_waveform.py -c ../config/tx_config.yaml -o ../plots/tx_dry_run.png
```

Optional: save generated complex samples:

```bash
python3 dry_run_waveform.py -c ../config/tx_config.yaml --save-npy ../plots/tx_dry_run.npy
```

Example outputs from the bundled A2G-500m campaign configs
(`config/a2g_500m/`):

| Variant                              | Dry-run preview                                  |
|--------------------------------------|--------------------------------------------------|
| Baseline 56 MHz, ZC-401 x 8          | ![baseline](figures/baseline_56mhz.png)          |
| High-resolution 100 MHz, ZC-401 x 8  | ![highres](figures/highres_100mhz.png)           |
| Long-sequence 56 MHz, ZC-1021 x 4    | ![longseq](figures/longseq_56mhz.png)            |

The "Configuration" panel reports range resolution, unambiguous delay, and
processing gain; the cross-correlation panel reports peak-to-sidelobe ratio
(PSLR) of the matched filter response.

## Configuration Guide

Both `config/tx_config.yaml` and `config/rx_config.yaml` use the same schema.

Key fields:

- `MODE`: `TX` or `RX`.
- `PERIOD`: burst scheduling rate (bursts per second).
- `WAVEFORM`: `ZC`, `PN`, or `CHIRP`.
- `USRP`: radio settings (`SERIAL`, `SAMPLE_RATE`, `CENTER_FREQ`, `GAIN`, clock/PPS refs).
- `RECV_OPTS`: RX duration, power/path-loss options, output type (`npz`/`mat`).
- `WAV_OPTS`: per-waveform parameters.
  For OFDM:
  `N_FFT`, `CP_LEN`, `SUBCARRIERS`, `N_PILOTS`, `DC_GUARD_BINS`, `EDGE_GUARD_BINS`, `SEED`, `NORMALIZE_PEAK`, `TARGET_PEAK`.
- `GPS`: optional USB GPS receiver (serial device, e.g. `/dev/ttyACM0`) for location/time metadata.

## Output

RX captures are written under:

- `measurements/YYYY-MM-DD_HH_MM/`

Each directory includes:

- `config.yaml`: config snapshot.
- `received_<timestamp>.npz` or `received_<timestamp>.mat`: measurement files.

## Post-processing

Install analysis dependencies:

```bash
pip install -r post_processing/requirements.txt
```

Use notebooks in `post_processing/`:

- `PostProcess.ipynb`
- `RX_review.ipynb`
- `SigMF_conv.ipynb`
- `SigMF_demo.ipynb`
- `Antenna.ipynb`

## Hardware Notes

- **USRP B210 / B205mini**: max usable bandwidth is 56 MHz over USB 3.0; the
  `highres_100mhz` campaign variant requires an X310-class radio.
- **Synchronization**: with 10 MHz + 1PPS (GPSDO or signal-generator)
  on TX and RX, ppm-level oscillator drift translates to sub-Hz frequency
  offset at 3.4 GHz, which is well below the inter-burst interval. Therefore, no
  runtime CFO correction is needed in the sounder.
- **DAC headroom**: each generated burst is rescaled so its peak magnitude
  equals `WAV_OPTS.TX_PEAK` (default 0.85, ~1.4 dB back-off). Without this,
  the unit-modulus ZC portion would sit at 0 dBFS for the entire sequence
  and any fixed-point rounding or analog ringing could clip the DAC.

## Dataset

Air-to-Ground Dryad dataset:

- https://datadryad.org/dataset/doi:10.5061/dryad.7h44j105p

## Citation

```bibtex
@inproceedings{gurses2024air,
  author    = {A. G{\"u}rses and M. L. Sichitiu},
  title     = {Air-to-Ground Channel Modeling for UAVs in Rural Areas},
  booktitle = {2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)},
  year      = {2024},
  pages     = {1--6},
  address   = {Washington, DC, USA},
  doi       = {10.1109/VTC2024-Fall63153.2024.10757825},
  publisher = {IEEE}
}
```

## Troubleshooting

- If `ChsRX` import fails, ensure `python3-dev`, `build-essential`, `cython`, and `numpy` are installed.
- If external/GPS lock never completes, verify clock/PPS cabling and matching `CLK_REF`/`PPS_REF` settings.
- Use `logs/out.log` for runtime diagnostics.
