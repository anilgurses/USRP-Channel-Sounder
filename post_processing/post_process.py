import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import plotly.express as px
import pandas as pd
from geopy.distance import geodesic
import plotly.io as pio
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from datetime import datetime, timedelta
import matplotlib as mpl
import simplekml
import yaml
import argparse
from tqdm.contrib.concurrent import process_map
from sigmf import SigMFFile
import commpy

from utils.antenna import *
from utils.vhcl_processor import * 
from utils.sig_processor import *
from utils.config_parser import *

PROCESS_OPTS = [
        "power_vs_dist",
        "power_vs_time",
        "pl_vs_dist",
        "pl_vs_time",
        "pl_vs_time_wsim",
        "pl_vs_dist_wsim"]

CC1 = (35.773851, -78.677010)
LW1 = (35.72747884, -78.69591754) 
LW2 = (35.72821230,-78.70093459)
H_TOWER_LW1 = 12
H_TOWER_LW2 = 10
UTC_TIME_OFF = 3
UTC_OFF_REQ = True
OFDM_LEN = 450
ZC_LEN = 2048
OFDM_GUARD = 300


class PostProcessor:
	def __init__(self) -> None:
		self.data = None

		# TODO change this to file path based antenna configuration read
		self.tx_ant = Antenna()
		self.rx_ant = Antenna()

		self.tx_ant.readTxAntenna()
		self.rx_ant.readRxAntenna()
		self.s_prc: SigProcessor = None # type: ignore
		self.exp_dir_list = [] 
		self.vhc_metrics = []
  
	# Arranging colors for the plots
	def get_colorscale(self):
		x = np.linspace(0, 1, 50)

		def list_rgba_colors(alpha=0.1):
			# colors = px.colors.get_colorscale("inferno")
			colors = sample_colorscale('jet', list(x))
			rgba_colors = [color.replace("rgb","rgba").replace(")", f", {alpha})") for color in colors]
			return rgba_colors

		opac_colors = list_rgba_colors(0.15) 
		fl_colors = list_rgba_colors(1.0)
		return opac_colors, fl_colors

	def _nearest(self, items, pivot):
		it = min(items, key=lambda x: abs(x - pivot))
		return items.index(it)

	def _find_closest_log(self, log_directory, input_date_str):
		# Parse the input date string
		input_date_format = "%Y-%m-%d_%H_%M"
		# Extract the date part from the path
		input_date_str = input_date_str.split('/')[-1]
		input_date = datetime.strptime(input_date_str, input_date_format)

		closest_log = None
		smallest_diff = None

		# Iterate through the files in the log directory
		for file_name in os.listdir(log_directory):
			log_path = os.path.join(log_directory, file_name)
			time_diff = None
			if file_name.endswith("_vehicleOut.txt"):
				try:
					file_date_str = file_name.split('_vehicleOut')[0]
					file_date = datetime.strptime(file_date_str, "%Y-%m-%d_%H_%M_%S")
					time_diff = abs((file_date - input_date).total_seconds())
				except (ValueError, IndexError) as e:
					print(f"Could not parse date from filename: {file_name}, error: {e}")
					continue
			elif file_name.endswith("_GPS.csv"):
				try:
					with open(log_path, 'r') as f:
						# Skip header
						f.readline()
						# Read the first data line
						first_line = f.readline()
						if first_line:
							timestamp_str = first_line.split(',')[0]
							log_start_time = datetime.fromtimestamp(float(timestamp_str))
							time_diff = abs((log_start_time - input_date).total_seconds())
				except (IOError, ValueError, IndexError) as e:
					print(f"Could not process file {file_name}: {e}")
					continue

			if time_diff is not None:
				if smallest_diff is None or time_diff < smallest_diff:
					closest_log = file_name
					smallest_diff = time_diff

		if closest_log:
			return os.path.join(log_directory, closest_log)
		return None

	def _parse_config(self, path):
		with open(path, "r") as f:
			self.config = yaml.load(f, Loader=yaml.FullLoader)
		return self.config

	def get_exp_dir(self, path, sigmf=False):
		for m in path:
			if sigmf:
				fls = glob.glob(m+"/*.sigmf-data")
			else:
				fls = glob.glob(m+"/*.npz")
			if len(fls) > 500:
				self.exp_dir_list.append(m)
		self.exp_dir_list.sort()

		return self.exp_dir_list

	def _conv_arr_to_df(self, arr):
		df = pd.DataFrame()
		for i in range(len(arr)):
			if arr[i]:
				if arr[i].detected:
					d_it = arr[i].__to_dict__()
					d_it["vehicle"] = pd.DataFrame.from_dict(arr[i].vehicle.__to_dict__(), orient="index")
					df = pd.concat([df, pd.DataFrame.from_dict(d_it, orient="index")], axis=1)

		return df.T

	def _df_type_corr(self, df):
		df["lat"] = df["lat"].astype(np.float32)
		df["lon"] = df["lon"].astype(np.float32)
		df["alt"] = df["alt"].astype(np.float32)
		df["dist"] = df["dist"].astype(np.float32)
		df["h_dist"] = df["h_dist"].astype(np.float32)
		df["v_dist"] = df["v_dist"].astype(np.float32)   

		df["avgPower"] = df["avgPower"].astype(np.float32)
		return df

	def _process_meas(self, path):
		r = np.load(path, allow_pickle=True) 
		measurement_time = datetime.fromtimestamp(r["time_info"].item())
		ind_n = self._nearest([v.time for v in self.vhc_metrics], measurement_time)
		rx_metric = self.vhc_metrics[ind_n]

		tx_metric = None
		if hasattr(self, 'tx_vhc_metrics') and self.tx_vhc_metrics:
			tx_ind_n = self._nearest([v.time for v in self.tx_vhc_metrics], measurement_time)
			tx_metric = self.tx_vhc_metrics[tx_ind_n]

		res = self.s_prc.process(r["rx_time"], r["rcv"][0], rx_metric, tx_metric)
		return res
	
	def _process_meas_sigmf(self, path):
		meta = sigmffile.fromfile(path.replace(".sigmf-data", ".sigmf-meta"))
		ts = datetime.fromtimestamp(meta.get_captures()[0]["core:timestamp"])
		rx_time = meta.get_captures()[0]["core:time"]
		rcv = np.fromfile(path, dtype=np.complex64)
		ind_n = self._nearest([v.time for v in self.vhc_metrics], ts)
		res = self.s_prc.process(rx_time, rcv, self.vhc_metrics[ind_n])
		return res

	def _pick_key_from_metrics(self, metrics, key):
		r = []
		for i in range(len(metrics)):
			if metrics[i]:
				if metrics[i].detected:
					r.append(getattr(metrics[i], key))
		return r

	def _process(self, path, sigmf=False, vhcl_log_path=None, tx_vhcl_log_path=None, interpolate_rate=1):
		if sigmf:
			measurements = sorted(glob.glob(path+"/*.sigmf-data"), key=os.path.getmtime)
			logs_dir = glob.glob(path)[0]
		else:
			measurements = sorted(glob.glob(path+"/*.npz"), key=os.path.getmtime)
			if vhcl_log_path:
				vhc_log = vhcl_log_path
			else:
				vhc_log = self._find_closest_log("../measurements/vehicle_logs/", path)
			logs_dir = glob.glob(vhc_log)[0]
			
		if not logs_dir:
			return None
		
		vehicle = Vehicle_Processor(self.config)
		vehicle.read_vehicle_data(logs_dir, sigmf)
		self.vhc_metrics = vehicle.get_metrics()
		print(f"Using vehicle log: {vhc_log} for processing.")
		
		if tx_vhcl_log_path:
			tx_vehicle = Vehicle_Processor(self.config)
			tx_vehicle.read_vehicle_data(tx_vhcl_log_path, sigmf)
			self.tx_vhc_metrics = tx_vehicle.get_metrics()

		local_zc_len = ZC_LEN
		if sigmf:
			meta = sigmffile.fromfile(measurements[0])
			if self.config.WAVEFORM == "ZC":
				local_zc_len = meta.get_global_info().get("core:zc_len", ZC_LEN)
				ROOT_IND = meta.get_global_info().get("core:zc_root_index", 0)
				print(f"ZC_LEN: {local_zc_len}, ROOT_IND: {ROOT_IND}")
				ref = commpy.zcsequence(
					ROOT_IND, local_zc_len
				)
		else:
			r = np.load(measurements[0], allow_pickle=True)
			ref = r["ref"]

		self.s_prc = SigProcessor(self.config, ref[:local_zc_len], None, local_zc_len*4, interpolate_rate=interpolate_rate)

		if sigmf:
			rp = process_map(self._process_meas_sigmf, measurements, max_workers=os.cpu_count())
		else:
			rp = process_map(self._process_meas, measurements, max_workers=os.cpu_count())

		return rp

	def process_date(self, path, process_force=False, sigmf=False, verbose=True, vhcl_log_path=None, tx_vhcl_log_path=None, interpolate_rate=1):
		res_dir = f"../field_data/post-results/{path.split('/')[-1]}/"

		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		
		if sigmf:
			meta = sigmffile.fromfile(glob.glob(path + "/*.sigmf-meta")[0])
			self.config = Config("", sigmf=True)
			self.config.sigmf_parser(meta)
		else:
			self.config = Config(path + "/config.yaml", sigmf=sigmf)
		prcsd = {
			"resultDir": [],
			"meas": [],
			"config": [],
			"freq": [],
			"waveType": []
		}
  
		if not os.path.exists(res_dir + "processed.pkl") or process_force:
			res = self._process(path, sigmf=sigmf, vhcl_log_path=vhcl_log_path, tx_vhcl_log_path=tx_vhcl_log_path, interpolate_rate=interpolate_rate)
			df = self._conv_arr_to_df(res)
			df = df.convert_dtypes()
			df = self._df_type_corr(df)
			df.reset_index(inplace=True)
			df.to_pickle(res_dir + "processed.pkl")

			freq = self.config.USRP_CONF.CENTER_FREQ / 1e6
			waveType = self.config.WAVEFORM

			prcsd["freq"].append(freq)
			prcsd["waveType"].append(waveType)
			prcsd["resultDir"].append(res_dir)
			prcsd["config"].append(self.config)
			prcsd["meas"].append(res)
			if verbose:
				print(f"Processing {path} measurement for the following experiments: \n")
		else:
			if sigmf:
				self.config.sigmf_parser(meta)
			else:
				self.config = Config(path + "/config.yaml", sigmf=sigmf)
			df = pd.read_pickle(res_dir + "processed.pkl")

			freq = self.config.USRP_CONF.CENTER_FREQ / 1e6
			waveType = self.config.WAVEFORM

			prcsd["freq"].append(freq)
			prcsd["waveType"].append(waveType)
			prcsd["resultDir"].append(res_dir)
			prcsd["config"].append(self.config)
			prcsd["meas"].append(df)
			if verbose:
				print(f"Mesaurement {path} already processed. Loading from cache.")
		print(f"{len(prcsd['meas'][0])} measurements from {path}.")
		self.pp_data = prcsd
		return prcsd

	def process_dates(self, path, process_force=False, sigmf=False, vhcl_log_path=None, tx_vhcl_log_path=None, interpolate_rate=1):
		self.get_exp_dir(path, sigmf=sigmf)
		prcsd = {
				"resultDir": [],
				"meas": [],
				"config": [],
				"freq": [],
				"waveType": []
		}
		with tqdm(self.exp_dir_list, desc="Processing measurements") as pbar:
			for m in pbar:
				res_dir = f"../field_data/post-results/{m.split('/')[-1]}/"
				if not os.path.exists(res_dir):
					os.makedirs(res_dir)  
    
				prc = self.process_date(m, process_force, sigmf=sigmf, vhcl_log_path=vhcl_log_path, tx_vhcl_log_path=tx_vhcl_log_path, interpolate_rate=interpolate_rate)
				prcsd["resultDir"].append(prc["resultDir"][0])
				prcsd["meas"].append(prc["meas"][0])
				prcsd["config"].append(prc["config"][0])
				prcsd["freq"].append(prc["freq"][0])
				prcsd["waveType"].append(prc["waveType"][0])
	
				if not os.path.exists(res_dir + "processed.pkl") or process_force:
					message = f"Processing {m} measurement for the following experiments: \n"
				else:
					message = f"Mesaurement {m} already processed. Loading from cache."
				pbar.set_postfix({"Status": message})
		# Post process data
		self.pp_data = prcsd
		return prcsd
	
	def plot_time_vs_power(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["time"], df["avgPower"])
		plt.xlabel("Time")
		plt.ylabel("Power")
		plt.title("Time vs Power")
		if save:
			plt.savefig("../field_data/post-results/time_vs_power.png")
		plt.show()
  
	def plot_multiple_time_vs_power(self, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		for i in range(len(self.pp_data["meas"])):
			df = self.pp_data["meas"][i]
			plt.plot(df["time"], df["avgPower"], label=f"Experiment {i}")
		plt.xlabel("Time")
		plt.ylabel("Power")
		plt.title("Time vs Power")
		plt.legend()
		if save:
			plt.savefig("../field_data/post-results/multiple_time_vs_power.png")
		plt.show()
  
	def plot_dist_vs_power(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["dist"], df["avgPower"])
		plt.xlabel("Distance")
		plt.ylabel("Power")
		plt.title("Distance vs Power")
		if save:
			plt.savefig("../field_data/post-results/dist_vs_power.png")
		plt.show()
  
	def plot_time_vs_loc(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		# Use plotly for map plotting
		df = self.pp_data["meas"][index]
		fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="avgPower", zoom=15)
		fig.update_layout(mapbox_style="open-street-map")
		fig.show()
		if save:
			fig.write_html("../field_data/post-results/time_vs_loc.html")
	
	def plot_multiple_time_vs_loc(self, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		# Use plotly for map plotting
		fig = go.Figure()
		for i in range(len(self.pp_data["meas"])):
			df = self.pp_data["meas"][i]
			fig.add_trace(go.Scattermapbox(
				lat=df["lat"],
				lon=df["lon"],
				mode="markers",
				marker=go.scattermapbox.Marker(
					size=14,
					color=df["avgPower"],
					colorscale="Jet",
					opacity=0.7
				),
				text=df["time"],
				name=f"Experiment {i}"
			))
		fig.update_layout(
			mapbox_style="open-street-map",
			mapbox=dict(
				center=go.layout.mapbox.Center(lat=35.773851, lon=-78.677010),
				zoom=10
			)
		)
		fig.show()
		if save:
			fig.write_html("../field_data/post-results/multiple_time_vs_loc.html")
   
	def plot_freqoff_vs_time(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["time"], df["freq_offset"])
		plt.xlabel("Time")
		plt.ylabel("Frequency Offset")
		plt.title("Frequency Offset vs Time")
		if save:
			plt.savefig("../field_data/post-results/freqoff_vs_time.png")
		plt.show()
  
	def plot_multiple_freqoff_vs_time(self, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		for i in range(len(self.pp_data["meas"])):
			df = self.pp_data["meas"][i]
			# Instead of experiment number, use frequency and average altitude
			plt.plot(df["time"], df["freq_offset"], label=f"{self.pp_data['freq'][i]} MHz, {math.floor(df['alt'].max())} m")
		plt.xlabel("Time")
		plt.ylabel("Frequency Offset")
		plt.title("Frequency Offset vs Time")
		plt.legend()
		if save:
			plt.savefig("../field_data/post-results/multiple_freqoff_vs_time.png")
		plt.show()
	
	def plot_snr_vs_time(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["time"], df["avgSnr"])
		plt.xlabel("Time")
		plt.ylabel("SNR (dB)")
		plt.title("SNR vs Time")
		if save:
			plt.savefig("../field_data/post-results/snr_vs_time.png")
		plt.show()
  
	def plot_multiple_snr_vs_time(self, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		for i in range(len(self.pp_data["meas"])):
			df = self.pp_data["meas"][i]
			plt.plot(df["time"], df["avgSnr"], label=f"Experiment {i}")
		plt.xlabel("Time")
		plt.ylabel("SNR (dB)")
		plt.title("SNR vs Time")
		plt.legend()
		if save:
			plt.savefig("../field_data/post-results/multiple_snr_vs_time.png")
		plt.show()

	def plot_doppler_spectrum(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		avg_doppler_spectrum = np.mean([s for s in df["doppler_spectrum"] if hasattr(s, '__len__') and len(s) > 1], axis=0)
		sample_rate = self.pp_data['config'][index].USRP_CONF.SAMPLE_RATE
		nperseg = 1024 # from sig_processor
		freq_axis = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/sample_rate))
		plt.plot(freq_axis, 10 * np.log10(np.fft.fftshift(avg_doppler_spectrum)))
		plt.xlabel("Frequency (Hz)")
		plt.ylabel("Power (dB)")
		doppler_shift = df["doppler_shift"].mean()
		plt.title(f"Doppler Spectrum (shift: {doppler_shift:.2f} Hz)")
		if save:
			plt.savefig("../field_data/post-results/doppler_spectrum.png")
		plt.show()

	def plot_multiple_doppler_spectrum(self, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		for i in range(len(self.pp_data["meas"])):
			df = self.pp_data["meas"][i]
			avg_doppler_spectrum = np.mean([s for s in df["doppler_spectrum"] if hasattr(s, '__len__') and len(s) > 1], axis=0)
			sample_rate = self.pp_data['config'][i].USRP_CONF.SAMPLE_RATE
			nperseg = 1024 # from sig_processor
			freq_axis = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/sample_rate))
			doppler_shift = df["doppler_shift"].mean()
			plt.plot(freq_axis, 10 * np.log10(np.fft.fftshift(avg_doppler_spectrum)), label=f"Experiment {i} (shift: {doppler_shift:.2f} Hz)")
		plt.xlabel("Frequency (Hz)")
		plt.ylabel("Power (dB)")
		plt.title("Doppler Spectrum")
		plt.legend()
		if save:
			plt.savefig("../field_data/post-results/multiple_doppler_spectrum.png")
		plt.show()

	def plot_rms_delay_spread_vs_dist(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]

		# Filter out non-positive values before taking log
		rms_delay_spread_ns = df["rms_delay_spread"] * 1e9
		valid_indices = rms_delay_spread_ns > 0

		plt.plot(df["dist"][valid_indices], np.log10(rms_delay_spread_ns[valid_indices]))
		plt.xlabel("Distance (m)")
		plt.ylabel("log10(RMS Delay Spread (ns))")
		plt.title("RMS Delay Spread vs Distance")
		if save:
			plt.savefig("../field_data/post-results/rms_delay_spread_vs_dist.png")
		plt.show()

	def plot_k_factor_vs_dist(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["dist"], df["k_factor"])
		plt.xlabel("Distance (m)")
		plt.ylabel("K-Factor (dB)")
		plt.title("K-Factor vs Distance")
		if save:
			plt.savefig("../field_data/post-results/k_factor_vs_dist.png")
		plt.show()

	def plot_rms_delay_spread_vs_time(self, index, save=False, offset=0):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]

		rms_delay_spread_ns = (df["rms_delay_spread"] + offset) * 1e9
		valid_indices = rms_delay_spread_ns > 0

		plt.plot(df["time"][valid_indices], np.log10(rms_delay_spread_ns[valid_indices]))
		plt.xlabel("Time")
		plt.ylabel("log10(RMS Delay Spread (ns))")
		plt.title("RMS Delay Spread vs Time")
		if save:
			plt.savefig("../field_data/post-results/rms_delay_spread_vs_time.png")
		plt.show()

	def plot_k_factor_vs_time(self, index, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		plt.plot(df["time"], df["k_factor"])
		plt.xlabel("Time")
		plt.ylabel("K-Factor (dB)")
		plt.title("K-Factor vs Time")
		if save:
			plt.savefig("../field_data/post-results/k_factor_vs_time.png")
		plt.show()
		
	def calculate_path_loss_exponent(self, index, plot=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return None, None

		df = self.pp_data["meas"][index]

		# Filter out invalid values
		df_filtered = df[df['avg_pl'].notna() & (df['dist'] > 1)] # Use dist > 1 to avoid log10(0) or negative
		
		if df_filtered.empty:
			print("No valid data to calculate path loss exponent.")
			return None, None

		# The model is PL = A + 10*n*log10(d)
		# We can rewrite this as PL = A + n * (10 * log10(d))
		# This is a linear equation y = c + m*x where:
		# y = PL (avg_pl)
		# x = 10 * log10(d)
		# m = n (path loss exponent)
		# c = A (path loss at 1m, or intercept)

		y = df_filtered['avg_pl']
		x = 10 * np.log10(df_filtered['dist'])

		coeffs = np.polyfit(x, y, 1)
		path_loss_exponent = coeffs[0]
		intercept = coeffs[1]
		
		print(f"Calculated Path Loss Exponent (n): {path_loss_exponent:.2f}")
		print(f"Intercept: {intercept:.2f} dB")
		
		if plot:
			self.plot_path_loss_fit(index, path_loss_exponent, intercept)
			
		return path_loss_exponent, intercept

	def plot_path_loss_fit(self, index, path_loss_exponent, intercept):
		df = self.pp_data["meas"][index]
		df_filtered = df[df['avg_pl'].notna() & (df['dist'] > 1)]
		
		plt.figure()
		plt.scatter(10 * np.log10(df_filtered['dist']), df_filtered['avg_pl'], label='Measured Data', alpha=0.5)
		
		# Create the fitted line
		x_fit_log = np.linspace(np.min(10 * np.log10(df_filtered['dist'])), np.max(10 * np.log10(df_filtered['dist'])), 100)
		y_fit = path_loss_exponent * x_fit_log + intercept
		
		plt.plot(x_fit_log, y_fit, color='red', linewidth=2, label=f'Fitted Line (n={path_loss_exponent:.2f})')
		
		plt.xlabel("10 * log10(Distance)")
		plt.ylabel("Path Loss (dB)")
		plt.title("Path Loss vs. Distance")
		plt.legend()
		plt.grid(True)
		plt.savefig("../field_data/post-results/path_loss_fit.png")
		plt.show()

	def plot_path_loss_exponent_vs_altitude(self, index, num_groups, save=False):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return

		df = self.pp_data["meas"][index]
		
		min_alt = df['alt'].min()
		max_alt = df['alt'].max()
		
		altitude_boundaries = np.linspace(min_alt, max_alt, num_groups + 1)
		
		ple_values = []
		altitude_centers = []

		for i in range(num_groups):
			lower_bound = altitude_boundaries[i]
			upper_bound = altitude_boundaries[i+1]
			
			if i == num_groups - 1:
				group_df = df[(df['alt'] >= lower_bound) & (df['alt'] <= upper_bound)]
			else:
				group_df = df[(df['alt'] >= lower_bound) & (df['alt'] < upper_bound)]

			# Filter out invalid values for path loss calculation
			df_filtered = group_df[group_df['avg_pl'].notna() & (group_df['dist'] > 1)]
			
			if len(df_filtered) < 2: # Need at least 2 points for a linear fit
				continue

			y = df_filtered['avg_pl']
			x = 10 * np.log10(df_filtered['dist'])
			
			# Perform linear regression
			coeffs = np.polyfit(x, y, 1)
			path_loss_exponent = coeffs[0]
			
			ple_values.append(path_loss_exponent)
			altitude_centers.append((lower_bound + upper_bound) / 2)

		if not ple_values:
			print("Could not calculate path loss exponent for any altitude group.")
			return

		plt.figure()
		plt.plot(altitude_centers, ple_values, marker='o', linestyle='-')
		
		plt.xlabel("Altitude (m)")
		plt.ylabel("Path Loss Exponent (n)")
		plt.title("Path Loss Exponent vs. Altitude")
		plt.grid(True)
		
		if save:
			plt.savefig("../field_data/post-results/path_loss_exponent_vs_altitude.png")
		plt.show()
    
	def plot_rms_delay_spread_cdf(self, index, save=False, offset=0):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		plt.figure()
		df = self.pp_data["meas"][index]
		
		# Remove NaN and inf values
		rms_delay_spread = df["rms_delay_spread"].dropna()
		rms_delay_spread = rms_delay_spread[np.isfinite(rms_delay_spread)]

		# Sort the data
		sorted_data_ns = (np.sort(rms_delay_spread) + offset) * 1e9

		# Filter out non-positive values
		valid_indices = sorted_data_ns > 0
		sorted_data_ns = sorted_data_ns[valid_indices]

		if len(sorted_data_ns) == 0:
			print("No positive RMS delay spread values to plot.")
			return
		
		# Calculate the CDF
		yvals = np.arange(len(sorted_data_ns)) / float(len(sorted_data_ns) - 1)
		
		plt.plot(np.log10(sorted_data_ns), yvals)
		plt.xlabel("log10(RMS Delay Spread (ns))")
		plt.ylabel("CDF")
		plt.title("RMS Delay Spread CDF")
		if save:
			plt.savefig("../field_data/post-results/rms_delay_spread_cdf.png")
		plt.show()

	def plot_rms_delay_spread_cdf_vs_altitude(self, index, num_groups, save=False, offset=0, use_log10_x_axis=True):
		if not self.pp_data:
			print("No processed data found. Please run process_date/s() first.")
			return
		
		plt.figure(figsize=(10, 6))
		
		df = self.pp_data["meas"][index]
		min_alt = df['alt'].min()
		max_alt = df['alt'].max()
		
		# Create altitude boundaries
		altitude_boundaries = np.linspace(min_alt, max_alt, num_groups + 1)
		
		for i in range(num_groups):
			lower_bound = altitude_boundaries[i]
			upper_bound = altitude_boundaries[i+1]
			
			# Filter the dataframe for the current altitude range
			# Make the upper bound inclusive for the last group
			if i == num_groups - 1:
				group_df = df[(df['alt'] >= lower_bound) & (df['alt'] <= upper_bound)]
			else:
				group_df = df[(df['alt'] >= lower_bound) & (df['alt'] < upper_bound)]
			
			if group_df.empty:
				continue
				
			# Get the rms_delay_spread data for the group
			rms_delay_spread = group_df["rms_delay_spread"].dropna()
			rms_delay_spread = rms_delay_spread[np.isfinite(rms_delay_spread)]
			
			if len(rms_delay_spread) == 0:
				continue
			
			# Sort the data and apply offset
			sorted_data_ns = (np.sort(rms_delay_spread) + offset) * 1e9  # Convert to ns
			valid_indices = sorted_data_ns > 0
			sorted_data_ns = sorted_data_ns[valid_indices]
			
			if len(sorted_data_ns) == 0:
				continue
			
			# Calculate the CDF properly for the filtered data
			cdf_values = np.arange(1, len(sorted_data_ns) + 1) / len(sorted_data_ns)
			
			label = f"{lower_bound:.0f} - {upper_bound:.0f} m"
			if use_log10_x_axis:
				x_values = np.log10(sorted_data_ns)
			else:
				x_values = sorted_data_ns
			
			plt.plot(x_values, cdf_values, label=label)
		
		if use_log10_x_axis:
			plt.xlabel("log10(RMS Delay Spread (ns))")
		else:
			plt.xlabel("RMS Delay Spread (ns)")
		plt.ylabel("CDF")
		plt.title("RMS Delay Spread CDF vs Altitude")
		plt.legend()
		plt.grid(False)
		plt.ylim([0, 1])

		# Make background transparent
		plt.gca().set_facecolor('none')
		plt.gcf().patch.set_alpha(0)

		if save:
			if use_log10_x_axis:
				save_path = "../field_data/post-results/rms_delay_spread_cdf_vs_altitude.png"
			else:
				save_path = "../field_data/post-results/rms_delay_spread_cdf_vs_altitude_linear_x.png"
			plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
		
		plt.show()
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", 
                        default="all",
                        const="all",
                        nargs="?",
                        choices=["power_vs_distance", "rms_delay_spread_vs_dist", "k_factor_vs_dist", "rms_delay_spread_vs_time", "k_factor_vs_time", "rms_delay_spread_cdf", "rms_delay_spread_cdf_vs_altitude", "path_loss_exponent", "path_loss_exponent_vs_altitude"])
    parser.add_argument("--sigmf", action="store_true")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset for RMS delay spread plot")
    parser.add_argument("--num_altitude_groups", type=int, help="Number of altitude groups for the CDF vs altitude plot")
    parser.add_argument("--interpolate_rate", type=int, default=1, help="Interpolation rate for higher resolution CIR")
    parser.add_argument("--linear_rms_cdf_x", action="store_true", help="Use linear x-axis for RMS delay spread CDF vs altitude plot")
    args = parser.parse_args()

    pp = PostProcessor()
    pp.process_dates(["../field_data/A2G_Channel_Measurements/2023-12-15_15_41/"], process_force=True, sigmf=args.sigmf, interpolate_rate=args.interpolate_rate)

    if args.generate == "rms_delay_spread_vs_dist":
        pp.plot_rms_delay_spread_vs_dist(0, save=True)
    elif args.generate == "k_factor_vs_dist":
        pp.plot_k_factor_vs_dist(0, save=True)
    elif args.generate == "rms_delay_spread_vs_time":
        pp.plot_rms_delay_spread_vs_time(0, save=True, offset=args.offset)
    elif args.generate == "k_factor_vs_time":
        pp.plot_k_factor_vs_time(0, save=True)
    elif args.generate == "rms_delay_spread_cdf":
        pp.plot_rms_delay_spread_cdf(0, save=True, offset=args.offset)
    elif args.generate == "rms_delay_spread_cdf_vs_altitude":
        if args.num_altitude_groups:
            pp.plot_rms_delay_spread_cdf_vs_altitude(
                0,
                args.num_altitude_groups,
                save=True,
                offset=args.offset,
                use_log10_x_axis=not args.linear_rms_cdf_x
            )
        else:
            print("Please provide the number of altitude groups using --num_altitude_groups")
    elif args.generate == "path_loss_exponent":
        pp.calculate_path_loss_exponent(0, plot=True)
    elif args.generate == "path_loss_exponent_vs_altitude":
        if args.num_altitude_groups:
            pp.plot_path_loss_exponent_vs_altitude(0, args.num_altitude_groups, save=True)
        else:
            print("Please provide the number of altitude groups using --num_altitude_groups")
