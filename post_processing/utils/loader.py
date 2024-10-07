import os
import sys
import datetime

@dataclass
class Location:
    lat: float 
    lon: float 
    alt: float
    pitch: float
    yaw: float
    roll: float
    vel_x: float 
    vel_y: float
    vel_z: float
    time: datetime 

class dataLoader:

    def __init__(self, path, config):
        self.loc_list = []

        measurements = sorted(glob.glob(path+"/*.npz"), key=os.path.getmtime)
        locs_dir = glob.glob(path + "/*_vehicleOut.txt") 
        

    def readLocTxt(self, path):
        with open(locs_dir[0], "r") as f:
            for l in f:
                loc = Location()
                ln = l.split(",")
                loc.lon = ln[1]
                loc.lat = ln[2]
                loc.alt = ln[3]
                locpitch = ln[4].replace('"(','')
                loc.yaw = ln[5]
                loc.roll = ln[6].replace(')"','')
                loc.vel_x = ln[7].replace('"(','')
                loc.vel_y = ln[8]
                loc.vel_z = ln[9].replace(')"','')
                loc.time = datetime.strptime(str(ln[11]), '%Y-%m-%d %H:%M:%S.%f')
                self.loc_list.append(loc) 
                
    def parseConfig(self, path):
        with open(path, "r") as stream:
            config = yaml.safe_load(stream)
        return config
    
    
