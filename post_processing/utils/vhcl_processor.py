from dataclasses import dataclass, asdict
from numpy import roll 
from datetime import datetime 
from sigmf import SigMFFile, sigmffile
import glob
import os
import pandas as pd
from geopy.distance import geodesic

@dataclass
class VehicleMetric:
    time: datetime
    type: str
    lat: float
    lon: float
    alt: float
    pitch: float
    yaw: float 
    roll: float
    vel_x: float
    vel_y: float
    vel_z: float
    heading: float = 0
    speed: float = 0
    
    def __post_init__(self):
        self.speed = (self.vel_x**2 + self.vel_y**2 + self.vel_z**2)**0.5
        
    def __str__(self) -> str:
        return f"VehicleMetric: time={self.time}, type={self.type}, lat={self.lat}, lon={self.lon}, alt={self.alt}, speed={self.speed}, heading={self.heading}, pitch={self.pitch}, yaw={self.yaw}, roll={self.roll}"
    
    def __repr__(self) -> str:
        return f"VehicleMetric: time={self.time}, type={self.type}, lat={self.lat}, lon={self.lon}, alt={self.alt}, speed={self.speed}, heading={self.heading}, pitch={self.pitch}, yaw={self.yaw}, roll={self.roll}"
    
    def __to_dict__(self):
        return asdict(self)
    
class Vehicle_Processor:
    
    def __init__(self, config) -> None:
        self.config = config
        self.metrics = []
      
    def parseMetrics(self, metrics):  
        v_metric = VehicleMetric(
            time=datetime.strptime(str(metrics[11]), '%Y-%m-%d %H:%M:%S.%f'),
            type="UAV",
            lon=float(metrics[1]),
            lat=float(metrics[2]),
            alt=float(metrics[3]),
            pitch=float(metrics[4].replace('"(','')),
            yaw=float(metrics[5]),
            roll=float(metrics[6].replace(')"','')),
            vel_x=float(metrics[7].replace('"(','')),
            vel_y=float(metrics[8]),
            vel_z=float(metrics[9].replace(')"',''))
        )
        return v_metric
    
    def parseMetricsSigmf(self, metrics):
        v_metric = VehicleMetric(
            time=datetime.fromtimestamp(metrics['core:timestamp']),
            type="UAV",
            lon=float(metrics['core:rx_location']['longitude']),
            lat=float(metrics['core:rx_location']['latitude']),
            alt=float(metrics['core:rx_location']['altitude']),
            pitch=float(metrics['core:rotation']['pitch']),
            yaw=float(metrics['core:rotation']['yaw']),
            roll=float(metrics['core:rotation']['roll']),
            vel_x=float(metrics['core:velocity']['velocity_x']),
            vel_y=float(metrics['core:velocity']['velocity_y']),
            vel_z=float(metrics['core:velocity']['velocity_z'])
        )
        return v_metric

    def parse_csv_metrics(self, row):
        v_metric = VehicleMetric(
            time=datetime.fromtimestamp(row['timestamp']),
            type="UAV",
            lon=float(row['Lng']),
            lat=float(row['Lat']),
            alt=float(row['Alt']),
            pitch=0.0, # Missing
            yaw=float(row['Yaw']),
            roll=0.0, # Missing
            vel_x=0.0, 
            vel_y=0.0,
            vel_z=float(row['VZ']),
            heading=float(row['GCrs']),
            speed=float(row['Spd'])
        )
        return v_metric

    def parse_ahr2_csv_metrics(self, row, vel_x, vel_y, vel_z):
        v_metric = VehicleMetric(
            time=datetime.fromtimestamp(row['timestamp']),
            type="UAV",
            lon=float(row['Lng']),
            lat=float(row['Lat']),
            alt=float(row['Alt']),
            pitch=float(row['Pitch']),
            yaw=float(row['Yaw']),
            roll=float(row['Roll']),
            vel_x=vel_x,
            vel_y=vel_y,
            vel_z=vel_z
        )
        return v_metric

    def read_vehicle_data(self, path, sigmf=False):
        if sigmf:
            files = glob.glob(os.path.join(path, '*.sigmf-meta'))
            if not files:
                print(f"No sigmf-meta files found in {path}")
                return
            for meta_file in files:
                meta = sigmffile.fromfile(meta_file).get_captures()[0]
                self.metrics.append(self.parseMetricsSigmf(meta))
        elif path.endswith('.csv'):
            try:
                df = pd.read_csv(path)
                if "AHR2" in path:
                    # Calculate time difference
                    df['time_diff'] = df['timestamp'].diff()
                    
                    # Calculate position differences
                    df['lat_diff'] = df['Lat'].diff()
                    df['lon_diff'] = df['Lng'].diff()
                    df['alt_diff'] = df['Alt'].diff()

                    # Calculate velocity (handle division by zero)
                    df['vel_x'] = df.apply(lambda row: (geodesic((row['Lat'] - row['lat_diff'], row['Lng']), (row['Lat'], row['Lng'])).meters) / row['time_diff'] if row['time_diff'] > 0 else 0, axis=1)
                    df['vel_y'] = df.apply(lambda row: (geodesic((row['Lat'], row['Lng'] - row['lon_diff']), (row['Lat'], row['Lng'])).meters) / row['time_diff'] if row['time_diff'] > 0 else 0, axis=1)
                    df['vel_z'] = df['alt_diff'] / df['time_diff']
                    
                    for index, row in df.iterrows():
                        self.metrics.append(self.parse_ahr2_csv_metrics(row, row['vel_x'], row['vel_y'], row['vel_z']))
                else:
                    for index, row in df.iterrows():
                        self.metrics.append(self.parse_csv_metrics(row))

            except ImportError:
                print("Pandas is not installed. Please install it to process CSV vehicle logs.")
            except Exception as e:
                print(f"Error processing vehicle log {path}: {e}")
        else:
            with open(path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    ln = line.split(',')
                    if len(ln) > 1:
                        self.metrics.append(self.parseMetrics(ln))
    
    def get_metrics(self):
        return self.metrics