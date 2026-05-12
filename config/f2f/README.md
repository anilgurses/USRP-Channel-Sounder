# Fixed-to-Fixed (F2F) Campaign Configs

5-node ground mesh at the Lake Wheeler testbed. Each run, one node transmits and the other four listen.

## Node coordinates

| Node | Lat | Lon | Alt (m) |
|------|-----|-----|---------|
| LW-1 | 35.72750947 | -78.69595819 | 12 |
| LW-2 | 35.72821305 | -78.70090823 | 10 |
| LW-3 | 35.72491205 | -78.69190014 | 20 |
| LW-4 | 35.73318358 | -78.69836420 | 12 |
| LW-5 | 35.74294142 | -78.69962993 | 18 |

Pairwise 3D distances: 454 m (lw1-lw2) up to 2119 m (lw3-lw5).

## Configs

| File | Role | Notes |
|------|------|-------|
| `longseq_56mhz_node.yaml` | Combined (TX+RX) | Single yaml per node, used in sweep mode. Coordinator assigns role per cycle. **Recommended.** |
| `longseq_56mhz_tx.yaml` / `longseq_56mhz_rx.yaml` | TX-only / RX-only | Legacy split for manual or single-shot captures. |

## Automated sweep (recommended)

```bash
# 1) Control host: start the coordinator in sweep mode.
python3 sounder/coordinator.py \
    --expected-nodes 5 \
    --sweep-tx-nodes lw1,lw2,lw3,lw4,lw5 \
    --sweep-subdevs A:A,A:B \      # B210/B205mini. For X310/N210 use A:0,B:0.
    --duration 30 \
    --lead-per-cycle-s 5
#    Writes `sweeps/sweep_<utc-iso>/manifest.json` as cycles complete.

# 2) Each node host (5 parallel SSH sessions, one config per node):
cd sounder/
python3 main.py -c ../config/f2f/longseq_56mhz_node.yaml \
    --coordinator <CTRL_HOST>:5555 --node-id lwN --tui
```

For every cycle, each node sees a `cycle_start` message with `tx_node`, `rx_subdev`, `channel_label`. Nodes matching `tx_node` run as TX (slot A); the rest run as RX on the assigned subdev. Per-cycle output lands at:

```
sounder/../measurements/sweep_<utc-iso>/cycle_<NN>_tx<lwX>_ch<A|B>/
    config.yaml
    received_*.npz   (one per burst; each carries `channel_label` and `tx_node`)
```

After the sweep finishes the coordinator broadcasts `shutdown` and `manifest.json` is finalised.

## Single-shot fallback

```bash
# Control host:
python3 sounder/coordinator.py --expected-nodes 5 --duration 30 --channel-label A
# Nodes:
python3 main.py -c ../config/f2f/longseq_56mhz_tx.yaml --coordinator <CTRL_HOST>:5555  # TX node
python3 main.py -c ../config/f2f/longseq_56mhz_rx.yaml --coordinator <CTRL_HOST>:5555  # RX nodes
```

For testing one node without a coordinator:

```bash
python3 main.py -c ../config/f2f/longseq_56mhz_rx.yaml --duration 30
```

Starts ~`INIT_DELAY` seconds (default 0.5 s) after launch and stops 30 s later. Omit `--duration` to run until SIGINT, or pass `--start-epoch <UTC float>` if you need to pin the start to a specific wall-clock moment (only needed when manually coordinating two nodes without the daemon).

## Post-processing (sweep)

```bash
# Gather every node's local measurements into one tree.
mkdir -p sweep_collected
for n in lw1 lw2 lw3 lw4 lw5; do
    rsync -a "${n}:/path/to/sounder/measurements/sweep_<utc-iso>/" "sweep_collected/${n}/sweep_<utc-iso>/"
done

# Also copy the coordinator manifest from the control host.
cp /path/to/sweeps/sweep_<utc-iso>/manifest.json sweep_collected/

# Process the whole sweep.
python3 post_processing/run_f2f_sweep.py \
    --manifest sweep_collected/manifest.json \
    --measurements-root sweep_collected/
```

Outputs:
- `results/measurements_lwN_<...>_chA/` and `..._chB/` per (TX, RX, channel)
- `results/comparison/f2f_*.{pdf,png}`

## Post-processing (single-shot)

```bash
# Stamp EXPERIMENT_TYPE/TX_NODE/RX_NODE into every captured config.yaml
python3 post_processing/tools/annotate_f2f_configs.py --schedule \
    "HH_MM:lw1,HH_MM:lw2,HH_MM:lw3,HH_MM:lw4,HH_MM:lw5"

# Process and generate figures
python3 post_processing/run_f2f.py --force
```
