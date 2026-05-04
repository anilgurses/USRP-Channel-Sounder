"""
SigMF read/write utility for the USRP channel sounder.

Metadata schema (custom extensions on top of SigMF core):
  global:
    core:waveform        - waveform type (ZC, PN, CHIRP, OFDM)
    core:zc_len          - ZC sequence length (ZC only)
    core:zc_root_index   - ZC root index     (ZC only)
    core:tx_gain_ref     - TX reference power (dBm)
    core:rx_gain_ref     - RX reference power (dBm)
  captures[0]:
    core:sample_start    - always 0 (required by SigMF spec)
    core:frequency       - center frequency (Hz)
    core:timestamp       - Unix epoch of the measurement (float)
    core:time            - relative time within the experiment (s)
    core:rx_location     - {latitude, longitude, altitude}
    core:tx_location     - {latitude, longitude, altitude}
    core:rotation        - {pitch, yaw, roll}  (radians)
    core:velocity        - {velocity_x, velocity_y, velocity_z}  (m/s)
    core:heading         - heading (degrees)
    core:flight_stage    - "Takeoff" | "Flight" | "Landing"
    core:speed           - scalar speed (m/s)
    core:dist            - 3-D distance TX-RX (m)

Stored signal:
  The .sigmf-data file contains the cross-correlation output (CIR estimate)
  as complex float 32 little-endian (cf32_le).  If the DataFrame still has
  a legacy 'cropped_sig' column (older cached results), that is used instead.
"""

import json
import os

import numpy as np
import sigmf
from sigmf import sigmffile

# SigMF datatype tag for complex float32 little-endian
_SIGMF_DTYPE = "cf32_le"
_AUTHOR = "https://aerpaw.org/"
_DESCRIPTION = "Air-to-Ground Channel Sounding Measurements"


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def read_meta(path: str):
    """
    Return a SigMFFile object for *path*.

    *path* may point to either the ``.sigmf-meta`` or ``.sigmf-data`` file.
    """
    meta_path = path.replace(".sigmf-data", ".sigmf-meta")
    return sigmffile.fromfile(meta_path)


def read_samples(path: str, dtype=np.complex64) -> np.ndarray:
    """
    Return the IQ samples stored in *path*.

    *path* may point to either the ``.sigmf-meta`` or ``.sigmf-data`` file.
    """
    data_path = path.replace(".sigmf-meta", ".sigmf-data")
    return np.fromfile(data_path, dtype=dtype)


def get_capture_info(meta) -> dict:
    """
    Extract a flat info dict from a SigMFFile object.

    Keys mirror the metadata schema described at the top of this module.
    """
    capture = meta.get_captures()[0]
    global_info = meta.get_global_info()
    return {
        "timestamp":      capture.get("core:timestamp"),
        "rx_time":        capture.get("core:time"),
        "frequency":      capture.get("core:frequency"),
        "sample_rate":    global_info.get("core:sample_rate"),
        "waveform":       global_info.get("core:waveform"),
        "zc_len":         global_info.get("core:zc_len"),
        "zc_root_index":  global_info.get("core:zc_root_index"),
        "tx_gain_ref":    global_info.get("core:tx_gain_ref"),
        "rx_gain_ref":    global_info.get("core:rx_gain_ref"),
        "rx_location":    capture.get("core:rx_location"),
        "tx_location":    capture.get("core:tx_location"),
        "rotation":       capture.get("core:rotation"),
        "velocity":       capture.get("core:velocity"),
        "flight_stage":   capture.get("core:flight_stage"),
        "speed":          capture.get("core:speed"),
        "dist":           capture.get("core:dist"),
        "heading":        capture.get("core:heading"),
    }


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _build_global(config) -> dict:
    global_meta = {
        "core:version":     sigmf.__version__,
        "core:datatype":    _SIGMF_DTYPE,
        "core:author":      _AUTHOR,
        "core:description": _DESCRIPTION,
        "core:sample_rate": config.USRP_CONF.SAMPLE_RATE,
        "core:num_channels": 1,
        "core:waveform":    config.WAVEFORM,
    }

    # Gain references – prefer config attributes, fall back to known defaults
    global_meta["core:tx_gain_ref"] = getattr(config, "TX_REF_DBM", 19.97)
    global_meta["core:rx_gain_ref"] = getattr(config, "RX_REF_DBM", -50.68)

    # Waveform-specific parameters
    if config.WAVEFORM == "ZC" and config.WAV_OPTS is not None:
        global_meta["core:zc_root_index"] = config.WAV_OPTS.ROOT_IND
        global_meta["core:zc_len"]        = config.WAV_OPTS.SEQ_LEN

    return global_meta


def _build_capture(meas, config, tx_location: dict) -> dict:
    timestamp = float(meas["vehicle"].iloc[0, 0].timestamp())
    return {
        "core:sample_start": 0,
        "core:frequency":    config.USRP_CONF.CENTER_FREQ,
        "core:timestamp":    timestamp,
        "core:time":         float(meas["time"]),
        "core:rx_location": {
            "latitude":  float(meas["lat"]),
            "longitude": float(meas["lon"]),
            "altitude":  float(meas["alt"]),
        },
        "core:tx_location": tx_location,
        "core:heading":      meas["heading"],
        "core:rotation": {
            "pitch": float(meas["pitch"]),
            "yaw":   float(meas["yaw"]),
            "roll":  float(meas["roll"]),
        },
        "core:velocity": {
            "velocity_x": float(meas["vel_x"]),
            "velocity_y": float(meas["vel_y"]),
            "velocity_z": float(meas["vel_z"]),
        },
        "core:flight_stage": meas["stage"],
        "core:speed":        float(meas["speed"]),
        "core:dist":         float(meas["dist"]),
    }


def _get_signal_data(meas) -> np.ndarray:
    """
    Return the complex signal array to write.

    Prefers the legacy 'cropped_sig' column for backwards compatibility with
    older cached DataFrames; otherwise uses the cross-correlation ('corr').
    """
    if "cropped_sig" in meas.index and meas["cropped_sig"] is not None:
        data = meas["cropped_sig"]
    elif "corr" in meas.index and meas["corr"] is not None and len(meas["corr"]) > 0:
        data = meas["corr"]
    else:
        return np.array([], dtype=np.complex64)

    return np.asarray(data, dtype=np.complex64)


def write_sigmf_file(meas, config, out_dir: str, tx_location: dict) -> None:
    """
    Write a single measurement row as a SigMF file pair.

    Args:
        meas:        A pandas Series row from the processed DataFrame.
        config:      The Config object for the experiment.
        out_dir:     Output directory (created if it does not exist).
        tx_location: Dict with 'latitude', 'longitude', 'altitude' for the TX.
    """
    os.makedirs(out_dir, exist_ok=True)

    timestamp = float(meas["vehicle"].iloc[0, 0].timestamp())
    base_name = f"Channel_Sounder_{timestamp}"
    data_path = os.path.join(out_dir, base_name + ".sigmf-data")
    meta_path = os.path.join(out_dir, base_name + ".sigmf-meta")

    metadata = {
        "global":      _build_global(config),
        "captures":    [_build_capture(meas, config, tx_location)],
        "annotations": [],
    }

    signal_data = _get_signal_data(meas)
    signal_data.tofile(data_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)


def write_sigmf_campaign(processed: dict, out_base_dir: str,
                         tx_location: dict = None) -> None:
    """
    Write all measurements from a processed campaign to SigMF file pairs.

    Args:
        processed:    Dict returned by PostProcessor.process_date() or
                      process_dates().
        out_base_dir: Base directory for output.  Each experiment is written
                      to ``<out_base_dir>/<experiment_date>/``.
        tx_location:  TX location dict.  If None, read from the first
                      result-directory name (falls back to LW1 / H_TOWER_LW1
                      from constants).
    """
    from utils.constants import LW1, H_TOWER_LW1  # local import to avoid circulars

    if tx_location is None:
        tx_location = {
            "latitude":  LW1[0],
            "longitude": LW1[1],
            "altitude":  H_TOWER_LW1,
        }

    n_items = len(processed["meas"])
    for i in range(n_items):
        result_dir = processed["resultDir"][i]
        out_dir = os.path.join(
            out_base_dir,
            os.path.basename(result_dir.rstrip("/"))
        )
        config = processed["config"][i]
        df = processed["meas"][i]
        for _, meas in df.iterrows():
            write_sigmf_file(meas, config, out_dir, tx_location)
        print(f"Written {len(df)} SigMF files to {out_dir}")
