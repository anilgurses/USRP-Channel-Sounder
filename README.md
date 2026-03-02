# USRP Channel Sounder

A configurable real-time channel sounder for USRP devices, with TX/RX runtime in `sounder/` and analysis tools in `post_processing/`.

## Repository Layout

- `sounder/`: runtime TX/RX code.
- `config/`: example configuration files.
- `post_processing/`: analysis scripts and notebooks.
- `requirements.txt`: runtime Python dependencies.
- `post_processing/requirements.txt`: analysis dependencies.

## Prerequisites

- Linux machine with UHD-compatible USRP hardware.
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

## Configuration Guide

Both `config/tx_config.yaml` and `config/rx_config.yaml` use the same schema.

Key fields:

- `MODE`: `TX` or `RX`.
- `PERIOD`: burst scheduling rate (bursts per second).
- `WAVEFORM`: `ZC`, `PN`, or `CHIRP`.
- `USRP`: radio settings (`SERIAL`, `SAMPLE_RATE`, `CENTER_FREQ`, `GAIN`, clock/PPS refs).
- `RECV_OPTS`: RX duration, power/path-loss options, output type (`npz`/`mat`).
- `WAV_OPTS`: per-waveform parameters.
- `GPS`: optional GNSS source for location/time metadata.

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

## Dataset

Dryad dataset:

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
