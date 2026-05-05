from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flatbuffers import encode, number_types, packer
from flatbuffers.table import Table

PA_ANOMALY_MIN_DELTA_DB = 3.0
EXTRAPOLATION_WARN_DELTA_MHZ = 5.0

REPO_ROOT = Path(__file__).resolve().parents[2]
CAL_DATA_DIR = REPO_ROOT / "config" / "cal_data"
TX_CONFIG_PATH = REPO_ROOT / "config" / "tx_config.yaml"
POWER_REFS_PATH = REPO_ROOT / "config" / "power_refs.csv"

DEFAULT_TX_REF_DBM = 15.0
DEFAULT_TX_REF_FREQ_HZ = 3.4e9
DEFAULT_TX_GAIN_DB = 64.0
DEFAULT_RX_REF_DBM = -48.83
DEFAULT_RX_GAIN_DB = 70.0

A2G_TX_NODE = "lw1"
A2G_RX_NODE = "pn6"
A2G_DEFAULT_CHANNEL = "ch0"
A2G_DEFAULT_TX_PA_ENABLED = 1
A2G_DEFAULT_RX_PA_ENABLED = 1


def _field_offset(tab: Table, field_id: int) -> int:
    return tab.Offset(4 + 2 * int(field_id))


def _indirect(buf: bytes, off: int) -> int:
    return off + encode.Get(packer.uoffset, buf, off)


def _root_table(buf: bytes) -> Table:
    return Table(buf, encode.Get(packer.uoffset, buf, 0))


def _get_scalar(tab: Table, field_id: int, flags, default=None):
    off = _field_offset(tab, field_id)
    if off == 0:
        return default
    return tab.Get(flags, tab.Pos + off)


def _get_string(tab: Table, field_id: int) -> str:
    off = _field_offset(tab, field_id)
    if off == 0:
        return ""
    raw = tab.String(tab.Pos + off)
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _get_table(tab: Table, field_id: int) -> Table | None:
    off = _field_offset(tab, field_id)
    if off == 0:
        return None
    return Table(tab.Bytes, _indirect(tab.Bytes, tab.Pos + off))


def _get_vector_tables(tab: Table, field_id: int) -> list[Table]:
    off = _field_offset(tab, field_id)
    if off == 0:
        return []
    vec = tab.Vector(off)
    n_items = tab.VectorLen(off)
    return [Table(tab.Bytes, _indirect(tab.Bytes, vec + i * 4)) for i in range(n_items)]


def _parse_tx_power_cal(path: Path) -> dict[str, object] | None:
    try:
        buf = path.read_bytes()
    except OSError:
        return None

    if len(buf) < 8 or buf[4:8] != b"dB/m":
        return None

    root = _root_table(buf)
    metadata = _get_table(root, 0)
    serial_full = _get_string(metadata, 1) if metadata is not None else path.stem
    ref_gain = _get_scalar(root, 2, number_types.Int32Flags, default=-1)

    rows: list[tuple[float, float, float]] = []
    for temp_map in _get_vector_tables(root, 1):
        for freq_map in _get_vector_tables(temp_map, 1):
            freq_hz = float(_get_scalar(freq_map, 0, number_types.Uint64Flags, default=0))
            off = _field_offset(freq_map, 1)
            if off == 0 or freq_hz <= 0:
                continue
            vec = freq_map.Vector(off)
            n_items = freq_map.VectorLen(off)
            for i in range(n_items):
                pos = vec + i * 16  # PowerMap struct: 2 doubles
                gain_db = float(freq_map.Get(number_types.Float64Flags, pos))
                power_dbm = float(freq_map.Get(number_types.Float64Flags, pos + 8))
                rows.append((freq_hz / 1e6, gain_db, power_dbm))

    if not rows:
        return None

    table = (
        pd.DataFrame(rows, columns=["freq_mhz", "gain_db", "power_dbm"])
        .groupby(["freq_mhz", "gain_db"], as_index=False)["power_dbm"]
        .mean()
        .pivot(index="freq_mhz", columns="gain_db", values="power_dbm")
        .sort_index()
        .sort_index(axis=1)
    )
    serial = serial_full.split("#", 1)[0]
    return {
        "path": str(path),
        "device_family": path.parent.name,
        "serial": serial,
        "serial_full": serial_full,
        "ref_gain": int(ref_gain),
        "table": table,
    }


@lru_cache(maxsize=4)
def load_tx_power_calibration_records(cal_dir: str | Path = CAL_DATA_DIR) -> tuple[dict[str, object], ...]:
    root = Path(cal_dir)
    records: list[dict[str, object]] = []
    if not root.exists():
        return tuple()

    for path in sorted(root.rglob("*.cal")):
        record = _parse_tx_power_cal(path)
        if record is not None:
            records.append(record)
    return tuple(records)


@lru_cache(maxsize=4)
def load_default_tx_settings(tx_config_path: str | Path = TX_CONFIG_PATH) -> dict[str, float | str]:
    path = Path(tx_config_path)
    defaults: dict[str, float | str] = {
        "serial": "",
        "gain_db": float(DEFAULT_TX_GAIN_DB),
        "center_freq_hz": float(DEFAULT_TX_REF_FREQ_HZ),
    }
    if not path.exists():
        return defaults

    try:
        with open(path, "r") as stream:
            cfg = yaml.safe_load(stream) or {}
    except Exception:
        return defaults

    usrp_cfg = cfg.get("USRP", {}) or {}
    defaults["serial"] = str(usrp_cfg.get("SERIAL", "") or "").strip()
    defaults["gain_db"] = float(usrp_cfg.get("GAIN", DEFAULT_TX_GAIN_DB))
    defaults["center_freq_hz"] = float(usrp_cfg.get("CENTER_FREQ", DEFAULT_TX_REF_FREQ_HZ))
    return defaults


def _parse_power_ref_freq_hz(column: str) -> float | None:
    name = str(column).strip()
    if not name.startswith("Freq_"):
        return None

    value = name[len("Freq_"):].strip()
    for suffix, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0)):
        if value.endswith(suffix):
            try:
                return float(value[:-len(suffix)]) * scale
            except ValueError:
                return None
    return None


@lru_cache(maxsize=4)
def load_power_reference_records(csv_path: str | Path = POWER_REFS_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    expected = {"Node", "Category", "Channel", "PA_Enabled", "Gain"}
    if not path.exists():
        return pd.DataFrame()

    try:
        table = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if not expected.issubset(table.columns):
        return pd.DataFrame()

    freq_columns = {
        col: _parse_power_ref_freq_hz(col)
        for col in table.columns
        if _parse_power_ref_freq_hz(col) is not None
    }
    if not freq_columns:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, row in table.iterrows():
        node = str(row.get("Node", "")).strip().lower()
        category = str(row.get("Category", "")).strip().lower()
        channel = str(row.get("Channel", "")).strip().lower()
        gain = pd.to_numeric(row.get("Gain"), errors="coerce")
        pa_enabled = pd.to_numeric(row.get("PA_Enabled"), errors="coerce")
        if not node or not category or not channel or not np.isfinite(gain):
            continue
        pa_value = int(pa_enabled) if np.isfinite(pa_enabled) else 0
        for col, freq_hz in freq_columns.items():
            power_dbm = pd.to_numeric(row.get(col), errors="coerce")
            if not np.isfinite(power_dbm):
                continue
            records.append({
                "node": node,
                "category": category,
                "channel": channel,
                "pa_enabled": pa_value,
                "gain_db": float(gain),
                "freq_hz": float(freq_hz),
                "power_dbm": float(power_dbm),
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def _path_contains_segment(path: str, segment: str) -> bool:
    needle = f"/{segment}/"
    return needle in path or path.endswith(f"/{segment}") or path.startswith(f"{segment}/") or path == segment


def infer_campaign_link_setup(campaign_path: str | Path | None) -> dict[str, object]:
    path = str(campaign_path or "").replace("\\", "/").lower().strip()
    if _path_contains_segment(path, "field_data/a2g") or _path_contains_segment(path, "a2g"):
        return {
            "experiment_type": "a2g",
            "tx_node": A2G_TX_NODE,
            "rx_node": A2G_RX_NODE,
            "tx_channel": A2G_DEFAULT_CHANNEL,
            "rx_channel": A2G_DEFAULT_CHANNEL,
            "tx_pa_enabled": A2G_DEFAULT_TX_PA_ENABLED,
            "rx_pa_enabled": A2G_DEFAULT_RX_PA_ENABLED,
        }
    if _path_contains_segment(path, "field_data/a2a") or _path_contains_segment(path, "a2a"):
        return {
            "experiment_type": "a2a",
            "tx_node": "",
            "rx_node": "",
            "tx_channel": "",
            "rx_channel": "",
            "tx_pa_enabled": 0,
            "rx_pa_enabled": 0,
        }
    return {
        "experiment_type": "",
        "tx_node": "",
        "rx_node": "",
        "tx_channel": "",
        "rx_channel": "",
        "tx_pa_enabled": 0,
        "rx_pa_enabled": 0,
    }


def apply_campaign_calibration_context(config, campaign_path: str | Path | None):
    inferred = infer_campaign_link_setup(campaign_path)
    defaults = load_default_tx_settings()

    if inferred["experiment_type"] and not getattr(config, "EXPERIMENT_TYPE", ""):
        config.EXPERIMENT_TYPE = str(inferred["experiment_type"])

    if inferred["tx_node"] and not getattr(config, "TX_NODE", ""):
        config.TX_NODE = str(inferred["tx_node"])
    if inferred["rx_node"] and not getattr(config, "RX_NODE", ""):
        config.RX_NODE = str(inferred["rx_node"])
    if inferred["tx_channel"] and not getattr(config, "TX_CHANNEL", ""):
        config.TX_CHANNEL = str(inferred["tx_channel"])
    if inferred["rx_channel"] and not getattr(config, "RX_CHANNEL", ""):
        config.RX_CHANNEL = str(inferred["rx_channel"])

    if not hasattr(config, "TX_PA_ENABLED"):
        config.TX_PA_ENABLED = int(inferred["tx_pa_enabled"])
    if not hasattr(config, "RX_PA_ENABLED"):
        config.RX_PA_ENABLED = int(inferred["rx_pa_enabled"])

    tx_gain_db = getattr(config, "TX_GAIN_DB", np.nan)
    try:
        tx_gain_valid = np.isfinite(float(tx_gain_db))
    except (TypeError, ValueError):
        tx_gain_valid = False
    if not tx_gain_valid:
        config.TX_GAIN_DB = float(defaults["gain_db"])

    if not getattr(config, "TX_SERIAL", ""):
        config.TX_SERIAL = str(defaults["serial"])
    if not getattr(config, "RX_SERIAL", ""):
        config.RX_SERIAL = str(getattr(config.USRP_CONF, "SERIAL", "") or "")

    return config


def _match_power_reference_subset(records: pd.DataFrame,
                                  *,
                                  node: str,
                                  category: str,
                                  channel: str,
                                  pa_enabled: int) -> tuple[pd.DataFrame, str]:
    if records.empty:
        return records, "no_records"

    filters = [
        (
            (records["node"] == node)
            & (records["category"] == category)
            & (records["channel"] == channel)
            & (records["pa_enabled"] == pa_enabled),
            "node_category_channel_pa",
        ),
        (
            (records["node"] == node)
            & (records["category"] == category)
            & (records["channel"] == channel),
            "node_category_channel",
        ),
        (
            (records["node"] == node)
            & (records["category"] == category)
            & (records["pa_enabled"] == pa_enabled),
            "node_category_pa",
        ),
        (
            (records["node"] == node)
            & (records["category"] == category),
            "node_category",
        ),
    ]

    for mask, selection in filters:
        subset = records.loc[mask].copy()
        if not subset.empty:
            return subset, selection
    return records.iloc[0:0].copy(), "no_match"


def resolve_power_reference_dbm(node: str,
                                category: str,
                                center_freq_hz: float,
                                gain_db: float,
                                *,
                                channel: str = "ch0",
                                pa_enabled: int = 0,
                                csv_path: str | Path = POWER_REFS_PATH,
                                policy: str = "linear") -> tuple[float, dict[str, object]]:
    node_key = str(node or "").strip().lower()
    category_key = str(category or "").strip().lower()
    channel_key = str(channel or "ch0").strip().lower()
    pa_value = int(pa_enabled)
    info = {
        "applied": False,
        "source": "power_refs_csv",
        "node": node_key,
        "category": category_key,
        "channel": channel_key,
        "pa_enabled": pa_value,
        "gain_db": float(gain_db),
        "target_freq_hz": float(center_freq_hz),
    }
    if not node_key or not category_key:
        info["reason"] = "missing_node_or_category"
        return float("nan"), info

    records = load_power_reference_records(csv_path)
    if records.empty:
        info["reason"] = "power_refs_missing_or_invalid"
        return float("nan"), info

    subset, selection = _match_power_reference_subset(
        records,
        node=node_key,
        category=category_key,
        channel=channel_key,
        pa_enabled=pa_value,
    )
    if subset.empty:
        info["reason"] = "no_matching_power_reference"
        return float("nan"), info

    info["selection"] = selection
    info["matched_channels"] = tuple(sorted(subset["channel"].unique()))
    info["matched_pa_enabled"] = tuple(sorted(int(v) for v in subset["pa_enabled"].unique()))

    grouped = (
        subset.groupby(["freq_hz", "gain_db"], as_index=False)["power_dbm"]
        .mean()
        .sort_values(["freq_hz", "gain_db"])
    )
    freq_points: list[float] = []
    power_points: list[float] = []
    for freq_hz, freq_group in grouped.groupby("freq_hz"):
        gains = freq_group["gain_db"].to_numpy(dtype=float)
        powers = freq_group["power_dbm"].to_numpy(dtype=float)
        valid = np.isfinite(gains) & np.isfinite(powers)
        if not np.any(valid):
            continue
        gains = gains[valid]
        powers = powers[valid]
        order = np.argsort(gains)
        gains = gains[order]
        powers = powers[order]
        if gains.size == 1:
            power_at_gain = float(powers[0])
        else:
            power_at_gain = float(np.interp(float(gain_db), gains, powers))
        freq_points.append(float(freq_hz))
        power_points.append(power_at_gain)

    if not freq_points:
        info["reason"] = "no_valid_interpolation_points"
        return float("nan"), info

    order = np.argsort(freq_points)
    freq_arr = np.asarray(freq_points, dtype=float)[order]
    power_arr = np.asarray(power_points, dtype=float)[order]
    target_freq_hz = float(center_freq_hz)
    if power_arr.size == 1:
        resolved = float(power_arr[0])
        extrapolated = True
        nearest_freq_hz = float(freq_arr[0])
        nearest_delta_mhz = abs(target_freq_hz - nearest_freq_hz) / 1e6
    else:
        extrapolated = bool(target_freq_hz < float(freq_arr[0]) or target_freq_hz > float(freq_arr[-1]))
        nearest_idx = int(np.argmin(np.abs(freq_arr - target_freq_hz)))
        nearest_freq_hz = float(freq_arr[nearest_idx])
        nearest_delta_mhz = abs(target_freq_hz - nearest_freq_hz) / 1e6
        if extrapolated:
            policy_norm = str(policy or "linear").strip().lower()
            if policy_norm == "fail":
                info["reason"] = "out_of_band"
                info["policy"] = policy_norm
                return float("nan"), info
            if policy_norm == "nearest":
                resolved = float(power_arr[nearest_idx])
            else:
                # Linear extrapolation from a least-squares fit through ALL
                # calibrated points (more stable than using just the last two).
                coeffs = np.polyfit(freq_arr, power_arr, 1)
                resolved = float(np.polyval(coeffs, target_freq_hz))
            info["policy"] = policy_norm
        else:
            resolved = float(np.interp(target_freq_hz, freq_arr, power_arr))
            info["policy"] = "interp"

    info.update({
        "applied": True,
        "freq_points_hz": tuple(float(v) for v in freq_arr),
        "resolved_ref_dbm": resolved,
        "extrapolated": bool(extrapolated),
        "nearest_calibrated_freq_hz": nearest_freq_hz,
        "nearest_calibrated_delta_mhz": float(nearest_delta_mhz),
    })
    return resolved, info


def resolve_rx_reference_dbm(center_freq_hz: float,
                             *,
                             rx_gain_db: float | None = None,
                             rx_serial: str | None = None,
                             rx_node: str | None = None,
                             rx_channel: str = "ch0",
                             rx_pa_enabled: int = 0) -> tuple[float, dict[str, object]]:
    gain_db = float(DEFAULT_RX_GAIN_DB if rx_gain_db is None else rx_gain_db)
    serial = str(rx_serial or "").strip()
    node = str(rx_node or "").strip().lower()

    if node:
        ref_dbm, info = resolve_power_reference_dbm(
            node,
            "rx",
            center_freq_hz,
            gain_db,
            channel=rx_channel,
            pa_enabled=rx_pa_enabled,
        )
        info = dict(info)
        info.update({
            "rx_serial": serial,
            "rx_node": node,
            "rx_channel": str(rx_channel),
            "rx_pa_enabled": int(rx_pa_enabled),
        })
        if np.isfinite(ref_dbm):
            return float(ref_dbm), info

    return float(DEFAULT_RX_REF_DBM), {
        "applied": False,
        "source": "fallback_constant",
        "reason": "missing_rx_node_power_reference",
        "rx_serial": serial,
        "rx_node": node,
        "rx_channel": str(rx_channel),
        "rx_pa_enabled": int(rx_pa_enabled),
        "gain_db": gain_db,
        "target_freq_hz": float(center_freq_hz),
        "resolved_ref_dbm": float(DEFAULT_RX_REF_DBM),
    }


def _select_tx_calibration_records(serial: str = "") -> tuple[tuple[dict[str, object], ...], str]:
    records = load_tx_power_calibration_records()
    if not records:
        return tuple(), "no_calibration_files"

    serial = str(serial or "").strip()
    if not serial:
        return records, "all_available"

    exact = tuple(
        rec for rec in records
        if str(rec["serial_full"]).strip() == serial or str(rec["serial"]).strip() == serial
    )
    if exact:
        return exact, "serial_exact"

    prefix = tuple(rec for rec in records if str(rec["serial"]).startswith(serial))
    if prefix:
        return prefix, "serial_prefix"

    return records, "serial_fallback_all"


def _interp_power_for_record(record: dict[str, object], freq_hz: float, gain_db: float) -> float:
    table = record["table"]
    if not isinstance(table, pd.DataFrame) or table.empty:
        return float("nan")

    target_freq_mhz = float(freq_hz) / 1e6
    gains = table.columns.to_numpy(dtype=float)
    if gains.size == 0:
        return float("nan")

    freq_points: list[float] = []
    power_points: list[float] = []
    for freq_mhz, row in table.iterrows():
        values = row.to_numpy(dtype=float)
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        freq_points.append(float(freq_mhz))
        power_points.append(float(np.interp(gain_db, gains[valid], values[valid])))

    if len(freq_points) < 2:
        return float("nan")

    return float(np.interp(target_freq_mhz, np.asarray(freq_points), np.asarray(power_points)))


def estimate_tx_port_power_dbm(freq_hz: float,
                               *,
                               gain_db: float | None = None,
                               serial: str | None = None) -> tuple[float, dict[str, object]]:
    settings = load_default_tx_settings()
    target_gain = float(settings["gain_db"] if gain_db is None else gain_db)
    target_serial = str(settings["serial"] if serial is None else serial).strip()

    records, selection = _select_tx_calibration_records(target_serial)
    estimates: list[float] = []
    used_records: list[str] = []
    for record in records:
        estimate = _interp_power_for_record(record, freq_hz, target_gain)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
            used_records.append(str(record["serial_full"]))

    info = {
        "applied": bool(estimates),
        "selection": selection,
        "tx_gain_db": target_gain,
        "tx_serial": target_serial,
        "n_cal_files": len(used_records),
        "matched_serials": tuple(used_records),
    }
    if not estimates:
        return float("nan"), info

    return float(np.mean(estimates)), info


def resolve_tx_reference_dbm(center_freq_hz: float,
                             *,
                             default_ref_dbm: float = DEFAULT_TX_REF_DBM,
                             tx_gain_db: float | None = None,
                             tx_serial: str | None = None,
                             reference_freq_hz: float | None = None,
                             tx_node: str | None = None,
                             tx_channel: str = "ch0",
                             tx_pa_enabled: int = 0) -> tuple[float, dict[str, object]]:
    settings = load_default_tx_settings()
    ref_freq_hz = float(settings["center_freq_hz"] if reference_freq_hz is None else reference_freq_hz)
    gain_db = float(settings["gain_db"] if tx_gain_db is None else tx_gain_db)
    serial = str(settings["serial"] if tx_serial is None else tx_serial).strip()
    node = str(tx_node or "").strip().lower()

    if node:
        ref_dbm, info = resolve_power_reference_dbm(
            node,
            "tx",
            center_freq_hz,
            gain_db,
            channel=tx_channel,
            pa_enabled=tx_pa_enabled,
        )
        info = dict(info)
        info.update({
            "applied": np.isfinite(ref_dbm),
            "default_ref_dbm": float(ref_dbm) if np.isfinite(ref_dbm) else float(default_ref_dbm),
            "resolved_ref_dbm": float(ref_dbm) if np.isfinite(ref_dbm) else float(default_ref_dbm),
            "reference_freq_hz": float(center_freq_hz),
            "target_freq_hz": float(center_freq_hz),
            "freq_correction_db": 0.0,
            "reference_port_power_dbm": float(ref_dbm) if np.isfinite(ref_dbm) else np.nan,
            "target_port_power_dbm": float(ref_dbm) if np.isfinite(ref_dbm) else np.nan,
            "tx_gain_db": gain_db,
            "tx_serial": serial,
            "tx_node": node,
            "tx_channel": str(tx_channel),
            "tx_pa_enabled": int(tx_pa_enabled),
        })
        if np.isfinite(ref_dbm):
            return float(ref_dbm), info

    ref_power_dbm, info = estimate_tx_port_power_dbm(ref_freq_hz, gain_db=gain_db, serial=serial)
    target_power_dbm, _ = estimate_tx_port_power_dbm(center_freq_hz, gain_db=gain_db, serial=serial)

    freq_correction_db = 0.0
    applied = bool(np.isfinite(ref_power_dbm) and np.isfinite(target_power_dbm))
    if applied:
        freq_correction_db = float(target_power_dbm - ref_power_dbm)

    resolved_dbm = float(default_ref_dbm + freq_correction_db)
    info = dict(info)
    info.update({
        "source": "uhd_calibration",
        "applied": applied,
        "default_ref_dbm": float(default_ref_dbm),
        "resolved_ref_dbm": resolved_dbm,
        "reference_freq_hz": ref_freq_hz,
        "target_freq_hz": float(center_freq_hz),
        "freq_correction_db": float(freq_correction_db),
        "reference_port_power_dbm": float(ref_power_dbm) if np.isfinite(ref_power_dbm) else np.nan,
        "target_port_power_dbm": float(target_power_dbm) if np.isfinite(target_power_dbm) else np.nan,
        "tx_node": node,
        "tx_channel": str(tx_channel),
        "tx_pa_enabled": int(tx_pa_enabled),
    })
    return resolved_dbm, info


def populate_link_budget_config(config, *, default_tx_ref_dbm: float = DEFAULT_TX_REF_DBM):
    settings = load_default_tx_settings()
    center_freq_hz = float(getattr(config.USRP_CONF, "CENTER_FREQ", settings["center_freq_hz"]))
    tx_gain_db = getattr(config, "TX_GAIN_DB", settings["gain_db"])
    rx_gain_db = getattr(config, "RX_GAIN_DB", getattr(config.USRP_CONF, "GAIN", DEFAULT_RX_GAIN_DB))

    try:
        tx_gain_db = float(tx_gain_db)
    except (TypeError, ValueError):
        tx_gain_db = float(settings["gain_db"])
    try:
        rx_gain_db = float(rx_gain_db)
    except (TypeError, ValueError):
        rx_gain_db = float(DEFAULT_RX_GAIN_DB)

    tx_serial = str(getattr(config, "TX_SERIAL", settings["serial"]) or settings["serial"]).strip()
    rx_serial = str(getattr(config, "RX_SERIAL", getattr(config.USRP_CONF, "SERIAL", "")) or "").strip()
    tx_node = str(getattr(config, "TX_NODE", "") or "").strip().lower()
    rx_node = str(getattr(config, "RX_NODE", "") or "").strip().lower()
    tx_channel = str(getattr(config, "TX_CHANNEL", "ch0") or "ch0").strip().lower()
    rx_channel = str(getattr(config, "RX_CHANNEL", "ch0") or "ch0").strip().lower()
    tx_pa_enabled = int(getattr(config, "TX_PA_ENABLED", 0) or 0)
    rx_pa_enabled = int(getattr(config, "RX_PA_ENABLED", 0) or 0)

    tx_ref_dbm, tx_cal = resolve_tx_reference_dbm(
        center_freq_hz,
        default_ref_dbm=default_tx_ref_dbm,
        tx_gain_db=tx_gain_db,
        tx_serial=tx_serial,
        tx_node=tx_node,
        tx_channel=tx_channel,
        tx_pa_enabled=tx_pa_enabled,
    )
    rx_ref_dbm, rx_cal = resolve_rx_reference_dbm(
        center_freq_hz,
        rx_gain_db=rx_gain_db,
        rx_serial=rx_serial,
        rx_node=rx_node,
        rx_channel=rx_channel,
        rx_pa_enabled=rx_pa_enabled,
    )

    config.TX_REF_DBM = float(tx_ref_dbm)
    config.TX_REF_DBM_NOMINAL = float(tx_ref_dbm if tx_cal.get("source") == "power_refs_csv" else default_tx_ref_dbm)
    config.TX_FREQ_CORRECTION_DB = float(tx_cal.get("freq_correction_db", 0.0))
    config.TX_GAIN_DB = float(tx_gain_db)
    config.TX_SERIAL = tx_serial
    config.TX_NODE = tx_node
    config.TX_CHANNEL = tx_channel
    config.TX_PA_ENABLED = tx_pa_enabled
    config.TX_REF_SOURCE = str(tx_cal.get("source", ""))
    config.TX_CALIBRATION = tx_cal

    config.RX_REF_DBM = float(rx_ref_dbm)
    config.RX_GAIN_DB = float(rx_gain_db)
    config.RX_SERIAL = rx_serial
    config.RX_NODE = rx_node
    config.RX_CHANNEL = rx_channel
    config.RX_PA_ENABLED = rx_pa_enabled
    config.RX_REF_SOURCE = str(rx_cal.get("source", ""))
    config.RX_CALIBRATION = rx_cal

    warnings_out: list[str] = []
    for side, info, node, channel, pa_enabled in (
        ("tx", tx_cal, tx_node, tx_channel, tx_pa_enabled),
        ("rx", rx_cal, rx_node, rx_channel, rx_pa_enabled),
    ):
        if not node:
            continue
        if info.get("extrapolated"):
            warnings_out.append(
                f"{side}_extrapolated:{node}@{center_freq_hz/1e6:.1f}MHz_from_{info.get('nearest_calibrated_freq_hz', 0.0)/1e6:.1f}MHz"
            )
        try:
            alt_ref, _ = resolve_power_reference_dbm(
                node,
                side,
                center_freq_hz,
                float(tx_gain_db if side == "tx" else rx_gain_db),
                channel=channel,
                pa_enabled=1 - int(pa_enabled),
            )
        except Exception:
            alt_ref = float("nan")
        this_ref = float(tx_ref_dbm if side == "tx" else rx_ref_dbm)
        if np.isfinite(alt_ref) and np.isfinite(this_ref):
            delta = abs(alt_ref - this_ref)
            if delta < PA_ANOMALY_MIN_DELTA_DB:
                msg = (
                    f"{side}_pa_anomaly:{node} PA={pa_enabled}->{1-int(pa_enabled)} "
                    f"differ by only {delta:.2f} dB (<{PA_ANOMALY_MIN_DELTA_DB:.0f} dB)"
                )
                logger.warning(msg)
                warnings_out.append(msg)

    config.LINK_BUDGET_WARNINGS = warnings_out
    return config
