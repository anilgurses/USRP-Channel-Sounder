import gc
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pandas as pd
from datetime import datetime
import argparse
from tqdm.contrib.concurrent import process_map
import commpy

from utils.antenna import *
from utils.antenna import AntennaCalibrator
from utils.vhcl_processor import *
from utils.sig_processor import *
from utils.config_parser import *
from utils.sigmf_handler import read_meta, read_samples, get_capture_info, write_sigmf_campaign
from utils.results import CampaignResult, CampaignCollection
from utils.plotting import PostProcessorPlots
from utils.pathloss import pathloss_fit_support
from utils.channel_models import ci_model_fit, fi_model_fit
from utils.usrp_calibration import (
	apply_campaign_calibration_context,
	populate_link_budget_config,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIELD_DATA_BASE = os.path.join(PROJECT_ROOT, "field_data")
# For air to ground data
A2G_MEAS_BASE = os.path.join(FIELD_DATA_BASE, "a2g")
# For air to air data
A2A_MEAS_BASE = os.path.join(FIELD_DATA_BASE, "a2a")
OLD_MEAS_BASE = os.path.join(FIELD_DATA_BASE, "measurements")
MEAS_BASES = [A2G_MEAS_BASE, A2A_MEAS_BASE, OLD_MEAS_BASE]
LEGACY_VEHICLE_LOGS_DIR = os.path.join(FIELD_DATA_BASE, "vehicle_logs")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results")
ZC_LEN           = 2048 # TODO This should be based on the configuration file, this is just a default value for now
EST_DIST_MAX_ERROR_M = 50.0
FREQ_OFFSET_EWMA_ALPHA = 0.1
FREQ_OFFSET_OUTLIER_MAD = 8.0

_SIGNAL_SCALAR_COLS = [
    "time", "center_freq", "dist", "h_dist", "v_dist", "wav_type", "detected",
    "avgPower", "avgSigPower", "avgSnr", "noise_power_dbfs", "freq_offset",
    "freq_offset_ewma",
    "avg_pl", "clip_frac", "clip_max", "saturation_status", "pl_valid",
    "tx_gain_db", "rx_gain_db", "avg_pl_antcal",
    "est_dist", "est_dist_raw",
    "start_point",
    "aod_theta", "aod_phi", "aoa_theta", "aoa_phi", "stage",
    "rsrp", "shadowing", "multipath", "delay",
    "doppler_shift", "rms_delay_spread", "k_factor",
]
_VEHICLE_SCALAR_COLS = [
    "lat", "lon", "alt", "pitch", "yaw", "roll",
    "vel_x", "vel_y", "vel_z", "heading", "speed",
]


class PostProcessor(PostProcessorPlots):
	def __init__(self) -> None:
		self.data    = None
		self.pp_data = None

		# TODO change this to file-path-based antenna configuration read
		self.tx_ant = Antenna()
		self.rx_ant = Antenna()
		self.tx_ant.readTxAntenna()
		self.rx_ant.readRxAntenna()
		self.antenna_calibrator = AntennaCalibrator(self.tx_ant, self.rx_ant)

		self.s_prc: SigProcessor = None  # type: ignore
		self.exp_dir_list = []
		self.vhc_metrics  = []
		self._measurement_file_cache = {}

	def _init_result_dirs(self, res_dir):
		for sub in [
			os.path.join(res_dir, "data"),
			os.path.join(res_dir, "figures"),
		]:
			os.makedirs(sub, exist_ok=True)

	@staticmethod
	def _category_filename(category, filename):
		stem = os.path.splitext(os.path.basename(filename))[0]
		if stem == category or stem.startswith(f"{category}_"):
			return filename
		return f"{category}_{filename}"

	@staticmethod
	def _prepare_output_file(path):
		if os.path.exists(path) and not os.path.isfile(path):
			raise IsADirectoryError(f"Output path exists and is not a file: {path}")
		if os.path.isfile(path) and not os.access(path, os.W_OK):
			os.unlink(path)
		return path

	def _fig_path(self, index, category, filename):
		base = os.path.join(self.pp_data["resultDir"][index], "figures")
		os.makedirs(base, exist_ok=True)
		path = os.path.join(base, self._category_filename(category, filename))
		return self._prepare_output_file(path)

	def _comparison_fig_path(self, category, filename):
		base = os.path.join(RESULTS_BASE, "comparison")
		os.makedirs(base, exist_ok=True)
		path = os.path.join(base, self._category_filename(category, filename))
		return self._prepare_output_file(path)

	def _measurement_sort_key(self, path, sigmf=False):
		base = os.path.basename(path)
		try:
			if sigmf:
				meta = read_meta(path)
				info = get_capture_info(meta)
				ts = info.get("timestamp", np.nan)
			else:
				with np.load(path, allow_pickle=True) as data:
					ts = float(data["time_info"].item())
			if np.isfinite(ts):
				return (0, float(ts), base)
		except Exception:
			pass
		return (1, float(os.path.getmtime(path)), base)

	def _ordered_measurement_files(self, campaign_path, sigmf=False, refresh=False):
		key = (os.path.abspath(campaign_path.rstrip("/")), bool(sigmf))
		if refresh or key not in self._measurement_file_cache:
			pattern = "*.sigmf-data" if sigmf else "*.npz"
			files = glob.glob(os.path.join(key[0], pattern))
			files.sort(key=lambda p: self._measurement_sort_key(p, sigmf=sigmf))
			self._measurement_file_cache[key] = files
		return self._measurement_file_cache[key]

	def _candidate_vehicle_log_dirs(self, campaign_path):
		"""Return possible vehicle-log directories for a campaign"""
		campaign_dir = os.path.abspath(campaign_path.rstrip("/"))
		parent = os.path.dirname(campaign_dir)
		candidates = [
			os.path.join(parent, "vehicle_logs"),
			LEGACY_VEHICLE_LOGS_DIR,
		]
		deduped = []
		seen = set()
		for item in candidates:
			item = os.path.abspath(item)
			if item not in seen and os.path.isdir(item):
				deduped.append(item)
				seen.add(item)
		return deduped

	def _resolve_vehicle_log_path(self, campaign_path, vhcl_log_path=None):
		"""best-matching vehicle log for a campaign"""
		if vhcl_log_path:
			return vhcl_log_path
		for log_dir in self._candidate_vehicle_log_dirs(campaign_path):
			match = self._find_closest_log(log_dir, campaign_path)
			if match:
				return match
		return None

	def _save_results(self, df, res_dir, config, delay_calibration=None,
	                  antenna_calibration=None, frequency_filter=None):
		data_dir = os.path.join(res_dir, "data")
		os.makedirs(data_dir, exist_ok=True)

		if antenna_calibration is None:
			antenna_calibration, cal_result = self._save_antenna_calibration(df, res_dir)
		else:
			cal_result = None

		if cal_result is not None and getattr(cal_result, "success", False):
			df = self.antenna_calibrator.apply(df, cal_result)
			if "pl_valid" in df.columns and "avg_pl_antcal" in df.columns:
				valid = pd.to_numeric(df["pl_valid"], errors="coerce").fillna(0).astype(bool)
				antcal_finite = pd.to_numeric(df["avg_pl_antcal"], errors="coerce").notna()
				df["pl_valid"] = (valid & antcal_finite).astype(bool)

		sig_cols = [c for c in _SIGNAL_SCALAR_COLS  if c in df.columns]
		vhc_cols = [c for c in _VEHICLE_SCALAR_COLS if c in df.columns]

		df[sig_cols].to_csv(os.path.join(data_dir, "signal_metrics.csv"),  index=False)
		df[vhc_cols].to_csv(os.path.join(data_dir, "vehicle_metrics.csv"), index=False)
		df[sig_cols + vhc_cols].to_csv(os.path.join(data_dir, "processed.csv"), index=False)

		summary = self._build_summary(
			df, config,
			delay_calibration=delay_calibration,
			antenna_calibration=antenna_calibration,
			frequency_filter=frequency_filter,
		)
		with open(os.path.join(data_dir, "summary.json"), "w") as fh:
			json.dump(summary, fh, indent=4)

	_PATTERN_OUTPUT_FILES = (
		("tx_pattern", "antenna_pattern_tx_binned.csv"),
		("rx_pattern", "antenna_pattern_rx_binned.csv"),
		("tx_samples", "antenna_pattern_tx_samples.csv"),
		("rx_samples", "antenna_pattern_rx_samples.csv"),
	)

	def _save_antenna_calibration(self, df, res_dir):
		data_dir = os.path.join(res_dir, "data")
		os.makedirs(data_dir, exist_ok=True)
		json_path = os.path.join(data_dir, "antenna_calibration.json")

		try:
			result = self.antenna_calibrator.fit(df)
		except Exception as exc:
			info = {
				"success": False,
				"message": f"antenna_calibration_failed: {exc}",
			}
			with open(json_path, "w") as fh:
				json.dump(info, fh, indent=4)
			return info, None

		for attr, filename in self._PATTERN_OUTPUT_FILES:
			pattern_df = getattr(result, attr, None)
			if pattern_df is not None:
				pattern_df.to_csv(os.path.join(data_dir, filename), index=False)

		info = result.to_dict()
		with open(json_path, "w") as fh:
			json.dump(info, fh, indent=4)
		return info, result

	@staticmethod
	def _populate_path_loss_calibration(config):
		populate_link_budget_config(config, default_tx_ref_dbm=TX_REF_DBM)
		return config

	@staticmethod
	def _path_loss_calibration_signature(config):
		return {
			"experiment_type": str(getattr(config, "EXPERIMENT_TYPE", "")),
			"tx_source": str(getattr(config, "TX_REF_SOURCE", "")),
			"tx_node": str(getattr(config, "TX_NODE", "")),
			"tx_channel": str(getattr(config, "TX_CHANNEL", "")),
			"tx_pa_enabled": int(getattr(config, "TX_PA_ENABLED", 0) or 0),
			"tx_gain_db": float(getattr(config, "TX_GAIN_DB", np.nan)),
			"tx_ref_dbm": float(getattr(config, "TX_REF_DBM", np.nan)),
			"rx_source": str(getattr(config, "RX_REF_SOURCE", "")),
			"rx_node": str(getattr(config, "RX_NODE", "")),
			"rx_channel": str(getattr(config, "RX_CHANNEL", "")),
			"rx_pa_enabled": int(getattr(config, "RX_PA_ENABLED", 0) or 0),
			"rx_gain_db": float(getattr(config, "RX_GAIN_DB", np.nan)),
			"rx_ref_dbm": float(getattr(config, "RX_REF_DBM", np.nan)),
		}

	@classmethod
	def _cache_requires_reprocess(cls, res_dir, config):
        """ Checking whether measurement needs to be processed or not 
        """
		summary_path = os.path.join(res_dir, "data", "summary.json")
		if not os.path.exists(summary_path):
			return True

		try:
			with open(summary_path, "r") as fh:
				summary = json.load(fh)
		except Exception:
			return True

		cached = summary.get("path_loss_calibration", {}) or {}

		cached_tx = str(cached.get("tx_node", "") or "").strip()
		cached_rx = str(cached.get("rx_node", "") or "").strip()
		if cached_tx in ("", "none") or cached_rx in ("", "none"):
			return True

		if "saturation" not in summary:
			return True

		current = cls._path_loss_calibration_signature(config)
		for key, expected in current.items():
			if key not in cached:
				return True
			actual = cached.get(key)
			if isinstance(expected, float):
				try:
					actual = float(actual)
				except (TypeError, ValueError):
					return True
				if not np.isfinite(expected) and not np.isfinite(actual):
					continue
				if not np.isfinite(actual) or abs(actual - expected) > 1e-6:
					return True
			elif actual != expected:
				return True
		return False

	def _load_results(self, res_dir):
		csv_path = os.path.join(res_dir, "data", "processed.csv")
		if not os.path.exists(csv_path):
			return None
		return pd.read_csv(csv_path)

	def _prepare_plotting_df(self, df, center_freq_hz):
		plot_df = df.copy()
		if "pl_valid" in plot_df.columns and "avg_pl" in plot_df.columns:
			valid = pd.to_numeric(plot_df["pl_valid"], errors="coerce").fillna(0).astype(bool)
			plot_df.loc[~valid, "avg_pl"] = np.nan
			if "avg_pl_antcal" in plot_df.columns:
				plot_df.loc[~valid, "avg_pl_antcal"] = np.nan
		return plot_df

	def _prepare_plotting_collection(self, collection):
		for result in collection:
			center_freq_hz = float(result.config.USRP_CONF.CENTER_FREQ)
			result.meas = self._prepare_plotting_df(result.meas, center_freq_hz)
		self.pp_data = collection
		return collection

	@staticmethod
	def _path_loss_plot_column(df, antenna_calibration=None):
		use_antcal = antenna_calibration is None
		if antenna_calibration is not None:
			confidence = str((antenna_calibration or {}).get("confidence", "")).strip().lower()
			use_antcal = confidence in {"medium", "high"}
		if use_antcal and "avg_pl_antcal" in df.columns:
			antcal = pd.to_numeric(df["avg_pl_antcal"], errors="coerce")
			if antcal.notna().any():
				return "avg_pl_antcal"
		return "avg_pl"

	def _path_loss_fit_frame(self, df, clean_only=True, airborne_only=True,
	                         antenna_calibration=None):
		pl_col = self._path_loss_plot_column(df, antenna_calibration=antenna_calibration)
		mask = pd.to_numeric(df.get(pl_col), errors="coerce").notna()
		mask &= pd.to_numeric(df.get("dist"), errors="coerce").gt(1.0)
		if "pl_valid" in df.columns:
			mask &= pd.to_numeric(df["pl_valid"], errors="coerce").fillna(0).astype(bool)
		if clean_only and "saturation_status" in df.columns:
			clean = df["saturation_status"].astype(str).eq("clean")
			if clean.any():
				mask &= clean
		if "stage" in df.columns:
			flight = df["stage"].eq("Flight")
			if flight.any():
				mask &= flight
		if airborne_only and "alt" in df.columns:
			alt = pd.to_numeric(df["alt"], errors="coerce")
			alt_valid = alt.dropna()
			if len(alt_valid):
				alt_target = float(np.nanpercentile(alt_valid, 90))
				if np.isfinite(alt_target) and alt_target > 0:
					airborne = alt > (0.8 * alt_target)
					if np.count_nonzero(mask & airborne.fillna(False)) >= 20:
						mask &= airborne.fillna(False)
		return df.loc[mask].copy()

	def _build_summary(self, df, config, delay_calibration=None,
	                   antenna_calibration=None, frequency_filter=None):
		def _stats(series):
			s = series.dropna()
			s = s[np.isfinite(s.values.astype(float))]
			if len(s) == 0:
				return {}
			return {
				"min":    float(np.min(s)),
				"p25":    float(np.percentile(s, 25)),
				"median": float(np.median(s)),
				"mean":   float(np.mean(s)),
				"p75":    float(np.percentile(s, 75)),
				"p90":    float(np.percentile(s, 90)),
				"max":    float(np.max(s)),
				"std":    float(np.std(s)),
			}

		def _masked_path_loss_stats(column):
			if column not in df.columns:
				return {}
			series = pd.to_numeric(df[column], errors="coerce").copy()
			if "pl_valid" in df.columns:
				valid = pd.to_numeric(df["pl_valid"], errors="coerce").fillna(0).astype(bool)
				series.loc[~valid] = np.nan
			return _stats(series)

		ple = None
		pl_intercept_db = None
		pl_fit = {
			"column": "",
			"n_samples": 0,
			"fit_supported": False,
			"reason": "insufficient_samples",
			"dist_q_span_m": np.nan,
			# Close-In (CI): 1m anchor fixed at FSPL(1m,f), one free param (PLE)
			"ci_path_loss_exponent": None,
			"ci_sigma_sf_db":        None,
			"ci_r_squared":          None,
			# Floating-Intercept (FI): two free params (alpha, 10*beta)
			"fi_alpha_db":           None,
			"fi_beta":               None,
			"fi_sigma_sf_db":        None,
			"fi_r_squared":          None,
			# kept for backward compat with downstream readers
			"intercept_db_1m":       None,
			"clean_only":  bool("saturation_status" in df.columns),
			"flight_only": bool("stage" in df.columns),
			"airborne_only": bool("alt" in df.columns),
		}
		try:
			pl_col = self._path_loss_plot_column(df, antenna_calibration=antenna_calibration)
			fit_df = self._path_loss_fit_frame(df, antenna_calibration=antenna_calibration)
			pl_fit["column"] = pl_col
			pl_fit["n_samples"] = int(len(fit_df))
			supported, fit_stats = pathloss_fit_support(fit_df["dist"])
			pl_fit["reason"] = str(fit_stats.get("reason", "unsupported"))
			pl_fit["dist_q_span_m"] = float(fit_stats.get("dist_q_span_m", np.nan))
			if supported:
				freq_hz = float(config.USRP_CONF.CENTER_FREQ)
				ci, fi = self._fit_path_loss_models(
					fit_df["dist"], fit_df[pl_col], freq_hz,
				)
				ple                       = ci["n"]
				pl_intercept_db           = fi["alpha_db"]
				pl_fit["ci_path_loss_exponent"] = ci["n"]
				pl_fit["ci_sigma_sf_db"]  = ci["sigma_sf_db"]
				pl_fit["ci_r_squared"]    = ci["r_squared"]
				pl_fit["fi_alpha_db"]     = fi["alpha_db"]
				pl_fit["fi_beta"]         = fi["beta"]
				pl_fit["fi_sigma_sf_db"]  = fi["sigma_sf_db"]
				pl_fit["fi_r_squared"]    = fi["r_squared"]
				pl_fit["intercept_db_1m"] = fi["alpha_db"]
				pl_fit["fit_supported"]   = True
				pl_fit["reason"]          = "ok"
		except Exception:
			pass

		stages = df["stage"].value_counts().to_dict() if "stage" in df.columns else {}

		sat_counts = {}
		n_clean = 0
		if "saturation_status" in df.columns and len(df):
			sat_counts = df["saturation_status"].astype(str).value_counts().to_dict()
			n_clean = int(sat_counts.get("clean", 0))
		clip_frac_stats = _stats(df["clip_frac"]) if "clip_frac" in df.columns else {}
		clip_max_stats = _stats(df["clip_max"]) if "clip_max" in df.columns else {}
		saturation = {
			"n_clean_frames":        n_clean,
			"n_total_frames":        int(len(df)),
			"clean_fraction":        float(n_clean / len(df)) if len(df) else 0.0,
			"status_counts":         {str(k): int(v) for k, v in sat_counts.items()},
			"clip_frac_p50":         float(clip_frac_stats.get("median", np.nan)) if clip_frac_stats else float("nan"),
			"clip_frac_p90":         float(clip_frac_stats.get("p90", np.nan)) if clip_frac_stats else float("nan"),
			"clip_max_p50":          float(clip_max_stats.get("median", np.nan)) if clip_max_stats else float("nan"),
			"clip_max_p90":          float(clip_max_stats.get("p90", np.nan)) if clip_max_stats else float("nan"),
		}

		return {
			"processed_at":    datetime.now().isoformat(),
			"n_measurements":  int(len(df)),
			"center_freq_mhz": float(config.USRP_CONF.CENTER_FREQ / 1e6),
			"waveform":        str(config.WAVEFORM),
			"altitude_m":             _stats(df["alt"]),
			"distance_m":             _stats(df["dist"]),
			"path_loss_db":           _masked_path_loss_stats("avg_pl"),
			"path_loss_antcal_db":    _masked_path_loss_stats("avg_pl_antcal"),
			"signal_power_dbfs":      _stats(df["avgSigPower"]) if "avgSigPower" in df.columns else {},
			"received_power_dbfs":    _stats(df["avgPower"]),
			"received_power_dbm":     _stats(df["avgPower"]),
			"snr_db":                 _stats(df["avgSnr"]),
			"freq_offset_hz":         _stats(df["freq_offset"]),
			"freq_offset_ewma_hz":    _stats(df["freq_offset_ewma"]) if "freq_offset_ewma" in df.columns else {},
			"rms_delay_spread_ns":    _stats(df["rms_delay_spread"] * 1e9),
			"k_factor_db":            _stats(df["k_factor"]),
			"path_loss_exponent":     ple,
			"path_loss_fit":          pl_fit,
			"flight_stages":          stages,
			"saturation":             saturation,
			"path_loss_valid_fraction": float(df["pl_valid"].mean()) if "pl_valid" in df.columns and len(df) else np.nan,
			"path_loss_calibration": {
				"experiment_type":      str(getattr(config, "EXPERIMENT_TYPE", "")),
				"tx_source":           str(getattr(config, "TX_REF_SOURCE", "")),
				"tx_ref_dbm":          float(getattr(config, "TX_REF_DBM", np.nan)),
				"tx_ref_nominal_dbm":  float(getattr(config, "TX_REF_DBM_NOMINAL", np.nan)),
				"tx_freq_correction_db": float(getattr(config, "TX_FREQ_CORRECTION_DB", 0.0)),
				"tx_gain_db":          float(getattr(config, "TX_GAIN_DB", np.nan)),
				"tx_serial":           str(getattr(config, "TX_SERIAL", "")),
				"tx_node":             str(getattr(config, "TX_NODE", "")),
				"tx_channel":          str(getattr(config, "TX_CHANNEL", "")),
				"tx_pa_enabled":       int(getattr(config, "TX_PA_ENABLED", 0) or 0),
				"tx_extrapolated":     bool(getattr(config, "TX_CALIBRATION", {}).get("extrapolated", False)) if isinstance(getattr(config, "TX_CALIBRATION", None), dict) else False,
				"rx_source":           str(getattr(config, "RX_REF_SOURCE", "")),
				"rx_ref_dbm":          float(getattr(config, "RX_REF_DBM", np.nan)),
				"rx_gain_db":          float(getattr(config, "RX_GAIN_DB", getattr(config.USRP_CONF, "GAIN", np.nan))),
				"rx_serial":           str(getattr(config, "RX_SERIAL", getattr(config.USRP_CONF, "SERIAL", ""))),
				"rx_node":             str(getattr(config, "RX_NODE", "")),
				"rx_channel":          str(getattr(config, "RX_CHANNEL", "")),
				"rx_pa_enabled":       int(getattr(config, "RX_PA_ENABLED", 0) or 0),
				"rx_extrapolated":     bool(getattr(config, "RX_CALIBRATION", {}).get("extrapolated", False)) if isinstance(getattr(config, "RX_CALIBRATION", None), dict) else False,
			},
			"link_budget_warnings":   list(getattr(config, "LINK_BUDGET_WARNINGS", []) or []),
			"delay_calibration":      delay_calibration or {},
			"frequency_offset_filter": frequency_filter or {},
			"antenna_calibration":    antenna_calibration or {},
		}

	# Utilities

	def _nearest(self, items, pivot):
		it = min(items, key=lambda x: abs(x - pivot))
		return items.index(it)

	@staticmethod
	def _fit_path_loss_models(dist, avg_pl, freq_hz):
		"""Fit Close-In (CI) and Floating-Intercept (FI) path-loss models.

		CI fixes the 1m intercept at FSPL(1m, freq_hz) and fits one parameter
		(the path-loss exponent n). FI fits two parameters (alpha_db at d=1m
		and beta = slope/10). 
		"""
		dist = np.asarray(dist, dtype=float)
		avg_pl = np.asarray(avg_pl, dtype=float)
		valid = np.isfinite(dist) & np.isfinite(avg_pl) & (dist > 1)
		if np.count_nonzero(valid) < 2:
			raise ValueError("Not enough samples for path-loss fitting.")
		d = dist[valid]
		pl = avg_pl[valid]
		ci_n, ci_sigma, ci_r2, _ = ci_model_fit(d, pl, float(freq_hz))
		fi_alpha, fi_beta, fi_sigma, fi_r2, _ = fi_model_fit(d, pl)
		return (
			{"n": float(ci_n), "sigma_sf_db": float(ci_sigma), "r_squared": float(ci_r2)},
			{
				"alpha_db":   float(fi_alpha),
				"beta":       float(fi_beta),
				"sigma_sf_db": float(fi_sigma),
				"r_squared":  float(fi_r2),
			},
		)

	@staticmethod
	def _calibrate_estimated_distance(df, sample_rate, max_error_m=EST_DIST_MAX_ERROR_M):
		if "start_point" not in df.columns or "dist" not in df.columns:
			return df, {"applied": False, "reason": "missing_columns"}

		valid = (
			df["start_point"].notna()
			& df["dist"].notna()
			& np.isfinite(df["start_point"])
			& np.isfinite(df["dist"])
			& (df["dist"] > 0)
		)
		if np.count_nonzero(valid) < 10:
			return df, {"applied": False, "reason": "insufficient_samples"}

		start_point = df.loc[valid, "start_point"].to_numpy(dtype=float)
		dist = df.loc[valid, "dist"].to_numpy(dtype=float)
		offsets = start_point - dist * sample_rate / SPEED_OF_LIGHT
		offset_median = float(np.median(offsets))
		offset_mad = float(np.median(np.abs(offsets - offset_median)))
		offset_sigma = 1.4826 * offset_mad

		info = {
			"applied": False,
			"offset_samples": offset_median,
			"robust_sigma_samples": float(offset_sigma),
			"n_samples": int(np.count_nonzero(valid)),
		}
		if not np.isfinite(offset_sigma) or offset_sigma > 100.0:
			info["reason"] = "unstable_offset"
			return df, info

		df = df.copy()
		if "est_dist" in df.columns:
			previous_est = pd.to_numeric(df["est_dist"], errors="coerce").to_numpy(dtype=float)
		else:
			previous_est = np.full(len(df), np.nan, dtype=float)
		if "est_dist_raw" not in df.columns and "est_dist" in df.columns:
			df["est_dist_raw"] = df["est_dist"]
		calibrated = np.maximum(
			0.0,
			(df["start_point"].to_numpy(dtype=float) - offset_median)
			* SPEED_OF_LIGHT / sample_rate,
		)
		gps_dist = pd.to_numeric(df["dist"], errors="coerce").to_numpy(dtype=float)
		est_valid = (
			np.isfinite(calibrated)
			& np.isfinite(gps_dist)
			& (gps_dist > 0.0)
			& (calibrated > 1.0)
			& (np.abs(calibrated - gps_dist) <= float(max_error_m))
		)
		df["est_dist"] = np.where(est_valid, calibrated, np.nan)
		info["applied"] = True
		info["max_error_m"] = float(max_error_m)
		info["n_valid_est_dist"] = int(np.count_nonzero(est_valid))
		info["n_discarded_est_dist"] = int(len(df) - np.count_nonzero(est_valid))
		info["changed"] = bool(
			previous_est.shape != df["est_dist"].to_numpy(dtype=float).shape
			or not np.allclose(
				previous_est,
				df["est_dist"].to_numpy(dtype=float),
				equal_nan=True,
			)
		)
		return df, info

	@staticmethod
	def _smooth_frequency_offset(df, alpha=FREQ_OFFSET_EWMA_ALPHA,
	                             outlier_mad=FREQ_OFFSET_OUTLIER_MAD):
		if "freq_offset" not in df.columns:
			return df, {"applied": False, "reason": "missing_freq_offset"}

		df = df.copy()
		raw = pd.to_numeric(df["freq_offset"], errors="coerce").to_numpy(dtype=float)
		previous = 
			pd.to_numeric(df["freq_offset_ewma"], errors="coerce").to_numpy(dtype=float)
			if "freq_offset_ewma" in df.columns
			else np.full(len(df), np.nan, dtype=float)
		)

		finite = np.isfinite(raw)
		filtered = raw.copy()
		n_outliers = 0
		if np.count_nonzero(finite) >= 5:
			median = float(np.nanmedian(raw[finite]))
			mad = float(np.nanmedian(np.abs(raw[finite] - median)))
			sigma = 1.4826 * mad
			if np.isfinite(sigma) and sigma > 0:
				outlier = finite & (np.abs(raw - median) > float(outlier_mad) * sigma)
				filtered[outlier] = np.nan
				n_outliers = int(np.count_nonzero(outlier))

		smoothed = (
			pd.Series(filtered)
			.ewm(alpha=float(alpha), adjust=False, ignore_na=True)
			.mean()
			.to_numpy(dtype=float)
		)
		df["freq_offset_ewma"] = smoothed
		info = {
			"applied": True,
			"method": "mad_filter_then_ewma",
			"alpha": float(alpha),
			"outlier_mad": float(outlier_mad),
			"n_samples": int(len(df)),
			"n_finite_raw": int(np.count_nonzero(finite)),
			"n_outliers": n_outliers,
			"changed": bool(
				previous.shape != smoothed.shape
				or not np.allclose(previous, smoothed, equal_nan=True)
			),
		}
		return df, info

	def _find_closest_log(self, log_directory, input_date_str):
		if not log_directory or not os.path.isdir(log_directory):
			return None
		input_date_format = "%Y-%m-%d_%H_%M"
		input_date_str = input_date_str.split("/")[-1]
		input_date = datetime.strptime(input_date_str, input_date_format)

		closest_log = None
		smallest_diff = None

		for file_name in os.listdir(log_directory):
			log_path  = os.path.join(log_directory, file_name)
			time_diff = None

			if file_name.endswith("_vehicleOut.txt"):
				try:
					file_date = datetime.strptime(file_name.split("_vehicleOut")[0],
					                              "%Y-%m-%d_%H_%M_%S")
					time_diff = abs((file_date - input_date).total_seconds())
				except (ValueError, IndexError) as e:
					print(f"Could not parse date from filename: {file_name}, error: {e}")
					continue

			elif file_name.endswith("_GPS.csv"):
				try:
					with open(log_path, "r") as f:
						f.readline()
						first_line = f.readline()
						if first_line:
							ts = datetime.fromtimestamp(float(first_line.split(",")[0]))
							time_diff = abs((ts - input_date).total_seconds())
				except (IOError, ValueError, IndexError) as e:
					print(f"Could not process file {file_name}: {e}")
					continue

			if time_diff is not None:
				if smallest_diff is None or time_diff < smallest_diff:
					closest_log   = file_name
					smallest_diff = time_diff

		return os.path.join(log_directory, closest_log) if closest_log else None

	def get_exp_dir(self, path, sigmf=False):
		for m in path:
			fls = glob.glob(m + ("/*.sigmf-data" if sigmf else "/*.npz"))
			if len(fls) > 500:
				self.exp_dir_list.append(m)
		self.exp_dir_list.sort()
		return self.exp_dir_list

	@staticmethod
	def _unwrap(v):
		if isinstance(v, np.ndarray) and v.ndim == 0:
			return v.item()
		return v

	def _df_type_corr(self, df):
		_f32 = ["lat", "lon", "alt", "dist", "h_dist", "v_dist",
		        "avgPower", "avgSigPower", "time", "aod_theta", "aod_phi", "aoa_theta", "aoa_phi",
		        "clip_frac", "clip_max"]
		_f64 = ["avgSnr", "freq_offset", "freq_offset_ewma", "avg_pl",
		        "noise_power_dbfs", "est_dist",
		        "doppler_shift", "rms_delay_spread", "k_factor",
		        "pitch", "yaw", "roll", "vel_x", "vel_y", "vel_z", "speed",
		        "center_freq"]
		_i64 = ["start_point", "rsrp", "shadowing", "multipath",
		        "delay", "heading", "pl_valid"]
		all_numeric = _f32 + _f64 + _i64
		for col in all_numeric:
			if col in df.columns:
				df[col] = df[col].apply(self._unwrap)
		for col in _f32:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
		for col in _f64:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
		for col in _i64:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
		# If it's string, leave it as it is
		for col in ["wav_type", "stage"]:
			if col in df.columns:
				df[col] = df[col].astype(str)
		return df

	def _process_meas(self, path):
		r = np.load(path, allow_pickle=True)
		measurement_time = datetime.fromtimestamp(r["time_info"].item())
		ind_n = self._nearest(self._vhc_times, measurement_time)
		rx_metric = self.vhc_metrics[ind_n]

		tx_metric = None
		if hasattr(self, "tx_vhc_metrics") and self.tx_vhc_metrics:
			tx_ind_n  = self._nearest(self._tx_vhc_times, measurement_time)
			tx_metric = self.tx_vhc_metrics[tx_ind_n]

		result = self.s_prc.process(r["rx_time"], r["rcv"][0], rx_metric, tx_metric, save_corr=False)
		if result is None or not result.detected:
			return None
		return result.to_scalar_dict()

	def _process_meas_sigmf(self, path):
		meta  = read_meta(path)
		info  = get_capture_info(meta)
		ts    = datetime.fromtimestamp(info["timestamp"])
		rcv   = read_samples(path)
		ind_n = self._nearest(self._vhc_times, ts)
		result = self.s_prc.process(info["rx_time"], rcv, self.vhc_metrics[ind_n], save_corr=False)
		if result is None or not result.detected:
			return None
		return result.to_scalar_dict()

	def _process(self, path, sigmf=False, vhcl_log_path=None,
	             tx_vhcl_log_path=None):

		measurements = self._ordered_measurement_files(path, sigmf=sigmf, refresh=True)
		if sigmf:
			logs_dir     = path
		else:
			vhc_log = self._resolve_vehicle_log_path(path, vhcl_log_path)
			if not vhc_log:
				candidates = self._candidate_vehicle_log_dirs(path)
				print(f"Warning: no vehicle log found for {path}. Checked: {candidates}")
				return None
			logs_dir = vhc_log

		vehicle = Vehicle_Processor(self.config)
		vehicle.read_vehicle_data(logs_dir, sigmf)
		self.vhc_metrics = vehicle.get_metrics()
		print(f"Vehicle log: {logs_dir}")
        
        # This is for if there are two vehicles
		if tx_vhcl_log_path:
			tx_vehicle = Vehicle_Processor(self.config)
			tx_vehicle.read_vehicle_data(tx_vhcl_log_path, sigmf)
			self.tx_vhc_metrics = tx_vehicle.get_metrics()

		local_zc_len = getattr(self.config.WAV_OPTS, "SEQ_LEN", ZC_LEN)
		if sigmf:
			meta = read_meta(measurements[0])
			info = get_capture_info(meta)
			if self.config.WAVEFORM == "ZC":
				local_zc_len = info["zc_len"] or ZC_LEN
				ROOT_IND     = info["zc_root_index"] or 0
				print(f"ZC_LEN: {local_zc_len}, ROOT_IND: {ROOT_IND}")
				ref = commpy.zcsequence(ROOT_IND, local_zc_len)
		else:
			r   = np.load(measurements[0], allow_pickle=True)
			ref = r["ref"]

		self.s_prc = SigProcessor(
			self.config, ref[:local_zc_len], None, local_zc_len * 4,
		)

		# Precompute vehicle time lists
		self._vhc_times = [v.time for v in self.vhc_metrics]
		if hasattr(self, 'tx_vhc_metrics') and self.tx_vhc_metrics:
			self._tx_vhc_times = [v.time for v in self.tx_vhc_metrics]

		fn = self._process_meas_sigmf if sigmf else self._process_meas
		n_workers  = os.cpu_count()
		chunksize  = max(1, len(measurements) // (n_workers * 4))
		results = process_map(fn, measurements, max_workers=n_workers, chunksize=chunksize)

		records = [r for r in results if r is not None]
		del results
		gc.collect()

		if not records:
			return None
		return pd.DataFrame(records)


	def reprocess_single_cir(self, index, meas_index, sigmf=None):
		if not self.pp_data:
			print("No processed data. Run process_date/s() first.")
			return None

		campaign = self.pp_data[index]
		campaign_path = campaign.campaign_path
		config = campaign.config

		if sigmf is None:
			sigmf = bool(glob.glob(os.path.join(campaign_path, "*.sigmf-data")))

		measurements = self._ordered_measurement_files(campaign_path, sigmf=sigmf)

		if meas_index < 0 or meas_index >= len(measurements):
			print(f"meas_index {meas_index} out of range (0..{len(measurements) - 1}).")
			return None

		meas_file = measurements[meas_index]

		# Ensure vehicle data and SigProcessor are initialised
		if self.s_prc is None or not self.vhc_metrics:
			# Bootstrap from the campaign
			local_zc_len = getattr(config.WAV_OPTS, "SEQ_LEN", ZC_LEN)
			if sigmf:
				meta = read_meta(measurements[0])
				info = get_capture_info(meta)
				if config.WAVEFORM == "ZC":
					local_zc_len = info["zc_len"] or ZC_LEN
					root_ind = info["zc_root_index"] or 0
					ref = commpy.zcsequence(root_ind, local_zc_len)
			else:
				r = np.load(measurements[0], allow_pickle=True)
				ref = r["ref"]

			self.s_prc = SigProcessor(
				config, ref[:local_zc_len], None, local_zc_len * 4,
			)

			vehicle = Vehicle_Processor(config)
			vehicle.read_vehicle_data(
				campaign_path if sigmf
				else self._resolve_vehicle_log_path(campaign_path),
				sigmf,
			)
			self.vhc_metrics = vehicle.get_metrics()
			self._vhc_times = [v.time for v in self.vhc_metrics]

		# Process the single measurement with save_corr=True
		if sigmf:
			meta = read_meta(meas_file)
			info = get_capture_info(meta)
			ts = datetime.fromtimestamp(info["timestamp"])
			rcv = read_samples(meas_file)
			ind_n = self._nearest(self._vhc_times, ts)
			result = self.s_prc.process(
				info["rx_time"], rcv, self.vhc_metrics[ind_n], save_corr=True,
			)
		else:
			r = np.load(meas_file, allow_pickle=True)
			measurement_time = datetime.fromtimestamp(r["time_info"].item())
			ind_n = self._nearest(self._vhc_times, measurement_time)

			tx_metric = None
			if hasattr(self, "tx_vhc_metrics") and self.tx_vhc_metrics:
				tx_ind_n = self._nearest(self._tx_vhc_times, measurement_time)
				tx_metric = self.tx_vhc_metrics[tx_ind_n]

			result = self.s_prc.process(
				r["rx_time"], r["rcv"][0], self.vhc_metrics[ind_n],
				tx_metric, save_corr=True,
			)

		if result is None or not result.detected:
			print(f"Measurement {meas_index} not detected.")
			return None
		return result

	def process_date(self, path, process_force=False, sigmf=False, verbose=True,
	                 vhcl_log_path=None, tx_vhcl_log_path=None):
		campaign_id = os.path.basename(path.rstrip("/"))
		res_dir     = os.path.join(RESULTS_BASE, campaign_id)
		self._init_result_dirs(res_dir)

		if sigmf:
			meta = read_meta(glob.glob(path + "/*.sigmf-meta")[0])
			self.config = Config("", sigmf=True)
			self.config.sigmf_parser(meta)
		else:
			self.config = Config(os.path.join(path, "config.yaml"), sigmf=sigmf)
		apply_campaign_calibration_context(self.config, path)
		self._populate_path_loss_calibration(self.config)

		prcsd = CampaignCollection()

		csv_path = os.path.join(res_dir, "data", "processed.csv")
		if os.path.exists(csv_path) and not process_force and self._cache_requires_reprocess(res_dir, self.config):
			process_force = True
			if verbose:
				print(f"Reprocessing {path}: cached A2G results were produced before lw1/pn6 power-ref calibration.")
		if not os.path.exists(csv_path) or process_force:
			df = self._process(
				path, sigmf=sigmf,
				vhcl_log_path=vhcl_log_path,
				tx_vhcl_log_path=tx_vhcl_log_path,
			)
			if df is None or df.empty:
				print(f"Skipping {path}: processing failed.")
				return prcsd

			df = self._df_type_corr(df)
			df.reset_index(inplace=True)
			df.sort_values("time", inplace=True, na_position="last")
			df.reset_index(drop=True, inplace=True)
			df, delay_calibration = self._calibrate_estimated_distance(
				df, self.config.USRP_CONF.SAMPLE_RATE
			)
			df, frequency_filter = self._smooth_frequency_offset(df)
			self._populate_path_loss_calibration(self.config)

			self._save_results(
				df, res_dir, self.config,
				delay_calibration=delay_calibration,
				frequency_filter=frequency_filter,
			)
			if verbose:
				print(f"Processed {path}  →  {res_dir}")
		else:
			if sigmf:
				meta = read_meta(glob.glob(path + "/*.sigmf-meta")[0])
				self.config.sigmf_parser(meta)
			else:
				self.config = Config(os.path.join(path, "config.yaml"), sigmf=sigmf)
			apply_campaign_calibration_context(self.config, path)

			df = self._load_results(res_dir)
			df.sort_values("time", inplace=True, na_position="last")
			df.reset_index(drop=True, inplace=True)
			df, delay_calibration = self._calibrate_estimated_distance(
				df, self.config.USRP_CONF.SAMPLE_RATE
			)
			df, frequency_filter = self._smooth_frequency_offset(df)
			self._populate_path_loss_calibration(self.config)
			antenna_cal_path = os.path.join(res_dir, "data", "antenna_calibration.json")
			if (
				not os.path.exists(antenna_cal_path)
				or delay_calibration.get("changed", False)
				or frequency_filter.get("changed", False)
			):
				# _save_results will compute antenna calibration, apply per-row
				# gain correction, and rewrite processed.csv + summary.json.
				self._save_results(
					df, res_dir, self.config,
					delay_calibration=delay_calibration,
					frequency_filter=frequency_filter,
				)
			if verbose:
				print(f"Loaded from cache: {csv_path}")

		freq     = self.config.USRP_CONF.CENTER_FREQ / 1e6
		waveType = self.config.WAVEFORM

		plot_df = self._prepare_plotting_df(df, self.config.USRP_CONF.CENTER_FREQ)

		prcsd.append(CampaignResult(
			result_dir=res_dir + "/",
			campaign_path=os.path.abspath(path),
			meas=plot_df,
			config=self.config,
			freq=freq,
			wave_type=waveType,
		))

		print(f"{len(df)} measurements from {path}.")
		self.pp_data = prcsd
		return prcsd

	def process_dates(self, path, process_force=False, sigmf=False,
	                  vhcl_log_path=None, tx_vhcl_log_path=None):
		self.get_exp_dir(path, sigmf=sigmf)
		prcsd = CampaignCollection()

		with tqdm(self.exp_dir_list, desc="Processing measurements") as pbar:
			for m in pbar:
				prc = self.process_date(
					m, process_force, sigmf=sigmf,
					vhcl_log_path=vhcl_log_path,
					tx_vhcl_log_path=tx_vhcl_log_path,
				)
				if not prc:
					continue
				prcsd.append(prc[0])
				pbar.set_postfix({"campaign": os.path.basename(m)})

		self._prepare_plotting_collection(prcsd)
		return prcsd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Channel Sounder Post-Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--campaign", metavar="DIR",
                     help="Path to a single campaign directory to process")
    src.add_argument("--date", metavar="YYYY-MM-DD",
                     help="Process all campaigns whose directory names start with this date")
    src.add_argument("--dir", metavar="DIR",
                     help="Process every campaign sub-directory inside DIR (e.g. ../field_data/a2g)")

    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if cached CSV exists")
    parser.add_argument("--sigmf", action="store_true",
                        help="Input files are in SigMF format instead of .npz")
    parser.add_argument("--altitude-groups", type=int, default=4, metavar="N",
                        help="Number of altitude groups for altitude-stratified plots")
    parser.add_argument("--figures", action="store_true",
                        help="Generate all figures after processing")
    parser.add_argument("--vhcl-log", metavar="PATH",
                        help="Override vehicle log path for the campaign")
    parser.add_argument("--tx-vhcl-log", metavar="PATH",
                        help="TX vehicle log for A2A mode (two moving aircraft)")
    parser.add_argument("--results-dir", metavar="DIR", default=None,
                        help="Override output results directory (default: ../results)")

    args = parser.parse_args()

    pp = PostProcessor()
    if args.results_dir:
        import sys
        sys.modules[__name__].RESULTS_BASE = args.results_dir

    if args.figures and "agg" in matplotlib.get_backend().lower():
        print(
            f"Figure generation is running in save-only mode (backend={matplotlib.get_backend()}). "
            f"Files are written under {RESULTS_BASE}; no live figure windows will open."
        )

    if args.campaign:
        pp.process_date(
            args.campaign,
            process_force=args.force,
            sigmf=args.sigmf,
            vhcl_log_path=args.vhcl_log,
            tx_vhcl_log_path=getattr(args, 'tx_vhcl_log', None),
        )
        if args.figures:
            pp.generate_all_figures(altitude_groups=args.altitude_groups)

    elif args.date:
        dirs = []
        for base in MEAS_BASES:
            if not os.path.isdir(base):
                continue
            dirs.extend(
                d for d in glob.glob(os.path.join(base, f"{args.date}*"))
                if os.path.isdir(d)
            )
        dirs = sorted(dict.fromkeys(dirs))
        if not dirs:
            print(f"No campaign directories found matching '{args.date}' in: {MEAS_BASES}")
        else:
            print(f"Found {len(dirs)} campaigns for {args.date}:")
            for d in dirs:
                print(f"  {d}")
            pp.process_dates(
                dirs,
                process_force=args.force,
                sigmf=args.sigmf,
            )
            if args.figures:
                pp.generate_all_figures(altitude_groups=args.altitude_groups)

    elif args.dir:
        dirs = sorted(
            d for d in glob.glob(os.path.join(args.dir, "*"))
            if os.path.isdir(d)
        )
        if not dirs:
            print(f"No sub-directories found in {args.dir}")
        else:
            print(f"Found {len(dirs)} campaign directories in {args.dir}:")
            for d in dirs:
                print(f"  {d}")
            pp.process_dates(
                dirs,
                process_force=args.force,
                sigmf=args.sigmf,
            )
            if args.figures:
                pp.generate_all_figures(altitude_groups=args.altitude_groups)
