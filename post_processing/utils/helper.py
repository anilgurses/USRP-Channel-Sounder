import os
from datetime import datetime

from sig_processor import *   
from vhcl_processor import * 
from config_parser import *


def nearest(items, pivot):
    it = min(items, key=lambda x: abs(x - pivot))
    return items.index(it) 

def find_closest_log(log_directory, input_date_str):
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

# @njit(parallel=True) 
def process_date(path, cfg, ext_loc=None):
    measurements = sorted(glob.glob(path+"/*.npz"), key=os.path.getmtime)
    vhc_log = find_closest_log("../field_data/vehicle_logs/",path)
    locs_dir = glob.glob(vhc_log) 

    if not locs_dir:
        return None
    
    print(path, locs_dir)

    vehicle = Vehicle_Processor(cfg)
    vehicle.read_vehicle_data(locs_dir[0])

    v_metrics = vehicle.get_metrics()
    
    r = np.load(measurements[0], allow_pickle=True) 

    s_prc = SigProcessor(cfg, r["ref"][:ZC_LEN], None, ZC_LEN*4)
    p_prc = PostProcessor(v_metrics, s_prc)
    
    # for m in tqdm(measurements):
    #     process_meas(m, vehicle, cfg, s_prc) 
 
    r = process_map(p_prc.process_meas, measurements, max_workers=os.cpu_count())
    # Parallel(n_jobs=20, require='sharedmem')(delayed(process_meas)(m, loc, d, cfg, tx_ant, rx_ant) for m in tqdm(measurements))
    return r

 def parseConfig(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config