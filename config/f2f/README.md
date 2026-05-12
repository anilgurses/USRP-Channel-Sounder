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

| File | Fs (MHz) | ZC SEQ_LEN | NUM_REPEATS | Range res | Unambig delay | Proc. gain | Notes |
|------|----------|-----------:|------------:|----------:|--------------:|-----------:|-------|
| `longseq_56mhz_*.yaml` | 56 | 1021 | 4 | 5.36 m | 27.4 us / 8.2 km | 30.1 dB | Primary F2F config |


## Running a measurement

Per-node command (RX or TX is selected by `MODE` inside the config):

The capture is now driven by a TCP **coordinator** that synchronises the start instant across all 5 nodes and stops every node after a configurable duration (default 30 s). See `sounder/coordinator.py`.

```bash
# 1) Control host: start the coordinator. It listens on :5555 until 5 nodes check in.
python3 sounder/coordinator.py \
    --expected-nodes 5 \
    --duration 30 \
    --channel-label A         # tags the per-node RX output directory

# 2) Each node host:
cd sounder/
# Transmitter:
python3 main.py -c ../config/f2f/longseq_56mhz_tx.yaml \
    --coordinator <CTRL_HOST>:5555 --tui
# Receivers (the other 4 nodes):
python3 main.py -c ../config/f2f/longseq_56mhz_rx.yaml \
    --coordinator <CTRL_HOST>:5555 --tui
```
