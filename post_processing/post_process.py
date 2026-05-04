import gc
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import pandas as pd
from datetime import datetime
import yaml
import argparse
from tqdm.contrib.concurrent import process_map
import commpy

from utils.antenna import *
from utils.vhcl_processor import *
from utils.sig_processor import *
from utils.config_parser import *
from utils.sigmf_handler import read_meta, read_samples, get_capture_info, write_sigmf_campaign
from utils.results import CampaignResult, CampaignCollection
from utils.plotting import PostProcessorPlots

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

_SIGNAL_SCALAR_COLS = [
    "time", "center_freq", "dist", "h_dist", "v_dist", "wav_type", "detected",
    "avgPower", "avgSnr", "freq_offset", "avg_pl", "est_dist", "start_point",
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

		self.s_prc: SigProcessor = None  # type: ignore
		self.exp_dir_list = []
		self.vhc_metrics  = []

	def _init_result_dirs(self, res_dir):
		for sub in [
			os.path.join(res_dir, "data"),
			os.path.join(res_dir, "figures", "power"),
			os.path.join(res_dir, "figures", "snr"),
			os.path.join(res_dir, "figures", "path_loss"),
			os.path.join(res_dir, "figures", "frequency"),
			os.path.join(res_dir, "figures", "delay_spread"),
			os.path.join(res_dir, "figures", "k_factor"),
			os.path.join(res_dir, "figures", "doppler"),
			os.path.join(res_dir, "figures", "distance"),
			os.path.join(res_dir, "figures", "cir"),
			os.path.join(res_dir, "figures", "kml"),
		]:
			os.makedirs(sub, exist_ok=True)

	def _fig_path(self, index, category, filename):
		base = os.path.join(self.pp_data["resultDir"][index], "figures", category)
		os.makedirs(base, exist_ok=True)
		return os.path.join(base, filename)

	def _comparison_fig_path(self, category, filename):
		base = os.path.join(RESULTS_BASE, "comparison", category)
		os.makedirs(base, exist_ok=True)
		return os.path.join(base, filename)

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

	def _save_results(self, df, res_dir, config):
		data_dir = os.path.join(res_dir, "data")
		os.makedirs(data_dir, exist_ok=True)

		sig_cols = [c for c in _SIGNAL_SCALAR_COLS  if c in df.columns]
		vhc_cols = [c for c in _VEHICLE_SCALAR_COLS if c in df.columns]

		df[sig_cols].to_csv(os.path.join(data_dir, "signal_metrics.csv"),  index=False)
		df[vhc_cols].to_csv(os.path.join(data_dir, "vehicle_metrics.csv"), index=False)
		df[sig_cols + vhc_cols].to_csv(os.path.join(data_dir, "processed.csv"), index=False)

		summary = self._build_summary(df, config)
		with open(os.path.join(data_dir, "summary.json"), "w") as fh:
			json.dump(summary, fh, indent=4)

	def _load_results(self, res_dir):
		csv_path = os.path.join(res_dir, "data", "processed.csv")
		if not os.path.exists(csv_path):
			return None
		return pd.read_csv(csv_path)

	def _build_summary(self, df, config):
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

		ple = None
		try:
			valid = df["avg_pl"].notna() & (df["dist"] > 1)
			ple, _ = self._fit_path_loss_model(df.loc[valid, "dist"],
			                                   df.loc[valid, "avg_pl"])
		except Exception:
			pass

		stages = df["stage"].value_counts().to_dict() if "stage" in df.columns else {}

		return {
			"processed_at":    datetime.now().isoformat(),
			"n_measurements":  int(len(df)),
			"center_freq_mhz": float(config.USRP_CONF.CENTER_FREQ / 1e6),
			"waveform":        str(config.WAVEFORM),
			"altitude_m":             _stats(df["alt"]),
			"distance_m":             _stats(df["dist"]),
			"path_loss_db":           _stats(df["avg_pl"]),
			"received_power_dbm":     _stats(df["avgPower"]),
			"snr_db":                 _stats(df["avgSnr"]),
			"freq_offset_hz":         _stats(df["freq_offset"]),
			"rms_delay_spread_ns":    _stats(df["rms_delay_spread"] * 1e9),
			"k_factor_db":            _stats(df["k_factor"]),
			"path_loss_exponent":     ple,
			"flight_stages":          stages,
		}

	# Utilities

	def _nearest(self, items, pivot):
		it = min(items, key=lambda x: abs(x - pivot))
		return items.index(it)

	@staticmethod
	def _fit_path_loss_model(dist, avg_pl):
		dist = np.asarray(dist, dtype=float)
		avg_pl = np.asarray(avg_pl, dtype=float)
		valid = np.isfinite(dist) & np.isfinite(avg_pl) & (dist > 1)
		if np.count_nonzero(valid) < 2:
			raise ValueError("Not enough samples for path-loss fitting.")
		log_d = np.log10(dist[valid])
		slope, intercept = np.polyfit(log_d, avg_pl[valid], 1)
		return float(slope / 10.0), float(intercept)

	@staticmethod
	def _latlon_to_local_xy(lat, lon, lat0=None, lon0=None):
		lat = np.asarray(lat, dtype=float)
		lon = np.asarray(lon, dtype=float)
		if lat0 is None:
			lat0 = float(np.nanmean(lat))
		if lon0 is None:
			lon0 = float(np.nanmean(lon))
		r_earth = 6_378_137.0
		x = np.deg2rad(lon - lon0) * r_earth * np.cos(np.deg2rad(lat0))
		y = np.deg2rad(lat - lat0) * r_earth
		return x, y, lat0, lon0

	def _build_reference_route(self, df):
		ref = df.sort_values("time").reset_index(drop=True)
		x_ref, y_ref, lat0, lon0 = self._latlon_to_local_xy(ref["lat"], ref["lon"])
		ref_pos = np.insert(np.cumsum(np.hypot(np.diff(x_ref), np.diff(y_ref))), 0, 0.0)
		return ref, x_ref, y_ref, ref_pos, lat0, lon0

	def _align_to_reference_route(self, df, x_ref, y_ref, ref_pos, lat0, lon0):
		aligned = df.sort_values("time").reset_index(drop=True).copy()
		x_tgt, y_tgt, _, _ = self._latlon_to_local_xy(aligned["lat"], aligned["lon"], lat0, lon0)
		aligned_pos = np.empty(len(aligned), dtype=float)
		chunk = 256
		for start in range(0, len(aligned), chunk):
			stop = min(start + chunk, len(aligned))
			dx = x_tgt[start:stop, None] - x_ref[None, :]
			dy = y_tgt[start:stop, None] - y_ref[None, :]
			nearest = np.argmin(dx * dx + dy * dy, axis=1)
			aligned_pos[start:stop] = ref_pos[nearest]
		aligned["aligned_pos_m"] = aligned_pos
		return aligned

	@staticmethod
	def _binned_track_profile(position_m, values, bin_width_m, min_count=3):
		pos = np.asarray(position_m, dtype=float)
		val = np.asarray(values, dtype=float)
		valid = np.isfinite(pos) & np.isfinite(val)
		if np.count_nonzero(valid) == 0:
			return np.array([]), np.array([])
		pos = pos[valid]
		val = val[valid]
		if pos.max() - pos.min() < bin_width_m:
			return np.array([np.median(pos)]), np.array([np.median(val)])
		bins = np.arange(pos.min(), pos.max() + bin_width_m, bin_width_m)
		centres = []
		stats = []
		inds = np.digitize(pos, bins) - 1
		for i in range(len(bins) - 1):
			in_bin = val[inds == i]
			if len(in_bin) < min_count:
				continue
			centres.append((bins[i] + bins[i + 1]) / 2.0)
			stats.append(np.median(in_bin))
		return np.asarray(centres, dtype=float), np.asarray(stats, dtype=float)

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

	def _parse_config(self, path):
		with open(path, "r") as f:
			self.config = yaml.load(f, Loader=yaml.FullLoader)
		return self.config

	def get_exp_dir(self, path, sigmf=False):
		for m in path:
			fls = glob.glob(m + ("/*.sigmf-data" if sigmf else "/*.npz"))
			if len(fls) > 500:
				self.exp_dir_list.append(m)
		self.exp_dir_list.sort()
		return self.exp_dir_list

	def _conv_arr_to_df(self, arr):
		records = []
		for item in arr:
			if item and item.detected:
				d = item.__to_dict__()
				v_dict = item.vehicle.__to_dict__()
				del v_dict["time"]
				del d["vehicle"]
				d.update(v_dict)
				records.append(d)
		return pd.DataFrame(records)

	@staticmethod
	def _unwrap(v):
		if isinstance(v, np.ndarray) and v.ndim == 0:
			return v.item()
		return v

	def _df_type_corr(self, df):
		_f32 = ["lat", "lon", "alt", "dist", "h_dist", "v_dist",
		        "avgPower", "time", "aod_theta", "aod_phi", "aoa_theta", "aoa_phi"]
		_f64 = ["avgSnr", "freq_offset", "avg_pl", "est_dist",
		        "doppler_shift", "rms_delay_spread", "k_factor",
		        "pitch", "yaw", "roll", "vel_x", "vel_y", "vel_z", "speed",
		        "center_freq"]
		_i64 = ["start_point", "rsrp", "shadowing", "multipath",
		        "delay", "heading"]
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

	# Low-level processing #

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

	def _pick_key_from_metrics(self, metrics, key):
		return [
			getattr(metrics[i], key)
			for i in range(len(metrics))
			if metrics[i] and metrics[i].detected
		]

	def _process(self, path, sigmf=False, vhcl_log_path=None,
	             tx_vhcl_log_path=None, interpolate_rate=1):

		if sigmf:
			measurements = sorted(glob.glob(path + "/*.sigmf-data"), key=os.path.getmtime)
			logs_dir     = path
		else:
			measurements = sorted(glob.glob(path + "/*.npz"), key=os.path.getmtime)
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

		local_zc_len = ZC_LEN
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
			self.config, ref[:local_zc_len], None,
			local_zc_len * 4, interpolate_rate=interpolate_rate
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

		if sigmf:
			measurements = sorted(
				glob.glob(os.path.join(campaign_path, "*.sigmf-data")),
				key=os.path.getmtime,
			)
		else:
			measurements = sorted(
				glob.glob(os.path.join(campaign_path, "*.npz")),
				key=os.path.getmtime,
			)

		if meas_index < 0 or meas_index >= len(measurements):
			print(f"meas_index {meas_index} out of range (0..{len(measurements) - 1}).")
			return None

		meas_file = measurements[meas_index]

		# Ensure vehicle data and SigProcessor are initialised
		if self.s_prc is None or not self.vhc_metrics:
			# Bootstrap from the campaign
			local_zc_len = ZC_LEN
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
				config, ref[:local_zc_len], None,
				local_zc_len * 4, interpolate_rate=1,
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
	                 vhcl_log_path=None, tx_vhcl_log_path=None, interpolate_rate=1):
		campaign_id = os.path.basename(path.rstrip("/"))
		res_dir     = os.path.join(RESULTS_BASE, campaign_id)
		self._init_result_dirs(res_dir)

		if sigmf:
			meta = read_meta(glob.glob(path + "/*.sigmf-meta")[0])
			self.config = Config("", sigmf=True)
			self.config.sigmf_parser(meta)
		else:
			self.config = Config(os.path.join(path, "config.yaml"), sigmf=sigmf)

		prcsd = CampaignCollection()

		csv_path = os.path.join(res_dir, "data", "processed.csv")
		if not os.path.exists(csv_path) or process_force:
			df = self._process(
				path, sigmf=sigmf,
				vhcl_log_path=vhcl_log_path,
				tx_vhcl_log_path=tx_vhcl_log_path,
				interpolate_rate=interpolate_rate,
			)
			if df is None or df.empty:
				print(f"Skipping {path}: processing failed.")
				return prcsd

			df = self._df_type_corr(df)
			df.reset_index(inplace=True)
			df.sort_values("time", inplace=True, na_position="last")
			df.reset_index(drop=True, inplace=True)

			self._save_results(df, res_dir, self.config)
			if verbose:
				print(f"Processed {path}  →  {res_dir}")
		else:
			if sigmf:
				meta = read_meta(glob.glob(path + "/*.sigmf-meta")[0])
				self.config.sigmf_parser(meta)
			else:
				self.config = Config(os.path.join(path, "config.yaml"), sigmf=sigmf)

			df = self._load_results(res_dir)
			df.sort_values("time", inplace=True, na_position="last")
			df.reset_index(drop=True, inplace=True)
			if verbose:
				print(f"Loaded from cache: {csv_path}")

		freq     = self.config.USRP_CONF.CENTER_FREQ / 1e6
		waveType = self.config.WAVEFORM

		prcsd.append(CampaignResult(
			result_dir=res_dir + "/",
			campaign_path=os.path.abspath(path),
			meas=df,
			config=self.config,
			freq=freq,
			wave_type=waveType,
		))

		print(f"{len(df)} measurements from {path}.")
		self.pp_data = prcsd
		return prcsd

	def process_dates(self, path, process_force=False, sigmf=False,
	                  vhcl_log_path=None, tx_vhcl_log_path=None, interpolate_rate=1):
		self.get_exp_dir(path, sigmf=sigmf)
		prcsd = CampaignCollection()

		with tqdm(self.exp_dir_list, desc="Processing measurements") as pbar:
			for m in pbar:
				prc = self.process_date(
					m, process_force, sigmf=sigmf,
					vhcl_log_path=vhcl_log_path,
					tx_vhcl_log_path=tx_vhcl_log_path,
					interpolate_rate=interpolate_rate,
				)
				if not prc:
					continue
				prcsd.append(prc[0])
				pbar.set_postfix({"campaign": os.path.basename(m)})

		self.pp_data = prcsd
		return prcsd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A2G Channel Sounder — Post-Processor",
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
    parser.add_argument("--interpolate-rate", type=int, default=1, metavar="N",
                        help="CIR interpolation factor (1 = no interpolation)")
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

    if args.campaign:
        pp.process_date(
            args.campaign,
            process_force=args.force,
            sigmf=args.sigmf,
            interpolate_rate=args.interpolate_rate,
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
                interpolate_rate=args.interpolate_rate,
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
                interpolate_rate=args.interpolate_rate,
            )
            if args.figures:
                pp.generate_all_figures(altitude_groups=args.altitude_groups)
