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
ZC_LEN = 401
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
		input_date_format = "%Y-%m-%d_%H_%M_%S"

		closest_log = None
		smallest_diff = None

		# Iterate through the files in the log directory
		for file_name in os.listdir(log_directory):
			if file_name.endswith("_vehicleOut.txt"):
				# Extract the date and time from the file name
				file_date_str = file_name.split('_vehicleOut')[0]
				file_date = datetime.strptime(file_date_str, input_date_format)

				# Calculate the time difference
				time_diff = abs((file_date - input_date).total_seconds())

				if smallest_diff is None or time_diff < smallest_diff:
					closest_log = file_name
					smallest_diff = time_diff

		return log_directory + closest_log

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
		ind_n = self._nearest([v.time for v in self.vhc_metrics], datetime.fromtimestamp(r["time_info"].item()))
		res = self.s_prc.process(r["rx_time"], r["rcv"][0], self.vhc_metrics[ind_n])
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

	def _process(self, path, sigmf=False):
		if sigmf:
			measurements = sorted(glob.glob(path+"/*.sigmf-data"), key=os.path.getmtime)
			logs_dir = glob.glob(path)[0]
		else:
			measurements = sorted(glob.glob(path+"/*.npz"), key=os.path.getmtime)
			vhc_log = self._find_closest_log("../field_data/vehicle_logs/", path)
			logs_dir = glob.glob(vhc_log)[0]

		if not logs_dir:
			return None

		vehicle = Vehicle_Processor(self.config)
		vehicle.read_vehicle_data(logs_dir, sigmf)
		self.vhc_metrics = vehicle.get_metrics()

		if sigmf:
			meta = sigmffile.fromfile(measurements[0])
			if self.config.WAVEFORM == "ZC":
				ZC_LEN = meta.get_global_info().get("core:zc_len", 401)
				ROOT_IND = meta.get_global_info().get("core:zc_root_index", 0)
				print(f"ZC_LEN: {ZC_LEN}, ROOT_IND: {ROOT_IND}")
				ref = commpy.zcsequence(
					ROOT_IND, ZC_LEN
				)
		else:
			r = np.load(measurements[0], allow_pickle=True)
			ref = r["ref"]

		self.s_prc = SigProcessor(self.config, ref[:ZC_LEN], None, ZC_LEN*4)

		if sigmf:
			rp = process_map(self._process_meas_sigmf, measurements, max_workers=os.cpu_count())
		else:
			rp = process_map(self._process_meas, measurements, max_workers=os.cpu_count())

		return rp

	def process_date(self, path, process_force=False, sigmf=False, verbose=True):
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
			res = self._process(path, sigmf=sigmf)
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

	def process_dates(self, path, process_force=False, sigmf=False):
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
    
				prc = self.process_date(m, process_force, False, sigmf=sigmf)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", 
                        default="all",
                        const="all",
                        nargs="?",
                        choices=["power_vs_distance", ""])
    parser.add_argument("--sigmf", action="store_true")
    args = parser.parse_args()

    pp = PostProcessor()
    pp.process_dates(["../field_data/A2G_Channel_Measurements/2023-12-15_15_41/"], process_force=True, sigmf=args.sigmf)

