import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import multiprocessing
import os
from tqdm import tqdm
from multiprocessing import Pool
import plotly.express as px
import pandas as pd
from geopy.distance import geodesic
import plotly.io as pio
import plotly.graph_objects as go
import glob
from datetime import datetime 
import matplotlib as mpl
import simplekml
from joblib import Parallel, delayed
import yaml
import argparse
from utils.antenna import *


PROCESS_OPTS = [
        "power_vs_dist",
        "power_vs_time",
        "pl_vs_dist", 
        "pl_vs_time", 
        "pl_vs_time_wsim", 
        "pl_vs_dist_wsim"]



class PostProcessor:
	def __init__(self, config) -> None:
		self.config = config
		self.data = None

		# TODO change this to file path based antenna configuration read
		self.tx_ant = Antenna()
		self.rx_ant = Antenna()
  
		self.tx_ant.readTxAntenna()
		self.rx_ant.readRxAntenna()
  

	def nearest(self, items, pivot):
		return min(items, key=lambda x: abs(x - pivot))


	def find_closest_log(self, log_directory, input_date_str):
		# Parse the input date string
		input_date_format = "%Y-%m-%d_%H_%M"
		input_date_str = input_date_str.split('/')[-1]  # Extract the date part from the path
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














if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", 
                        default="all",
                        const="all",
                        nargs="?",
                        choices=["power_vs_distance", ""])
    args = parser.parse_args()

