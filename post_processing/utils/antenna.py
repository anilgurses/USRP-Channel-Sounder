import numpy as np 
from scipy.interpolate import griddata
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv 
from geographiclib.geodesic import Geodesic


class Antenna:

    def __init__(self) -> None:
        self.phi = []
        self.theta = []
        self.gain_total = []
        self.gain_total_db = []
        self.gain_theta = []
        self.gain_theta_db = []
        self.gain_phi = []
        self.gain_phi_db = []
        
        self.df = pd.DataFrame(columns=['phi', 'theta', 'gain_total', 'gain_total_db', 'gain_theta', 'gain_theta_db', 'gain_phi', 'gain_phi_db'])

    @classmethod
    def llh_to_ecef(cls, lat, lon, alt):
        # WGS84 ellipsoid constants
        a = 6378137  # semi-major axis in meters
        f = 1 / 298.257223563  # flattening
        e_sq = f * (1 - f) **2  # square of eccentricity

        # Convert latitude and longitude to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Calculate the radius of curvature in the prime vertical
        N = a / math.sqrt(1 - e_sq * math.sin(lat_rad)**2)

        # Calculate Cartesian coordinates
        X = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        Y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        Z = (N * (1 - e_sq) + alt) * math.sin(lat_rad)

        return X, Y, Z

    @classmethod 
    def get_elevation_angle(cls, lat1, lon1, lat2, lon2, h1, h2):
        # Calculates the elevation angle between two points
        # lat1, lon1: coordinates of the first point
        # lat2, lon2: coordinates of the second point
        # h1, h2: height of the first and second point
        # Returns the elevation angle in radians
        x1, y1, z1 = Antenna.llh_to_ecef(lat1, lon1, h1) 
        x2, y2, z2 = Antenna.llh_to_ecef(lat2, lon2, h2)
        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2-z1)**2)
        h = z2 - z1
        return np.arctan(h/d)
    
    @classmethod
    def get_azimuth_angle(cls, lat1, lon1, lat2, lon2 ):
        # Calculates the azimuth angle between two points
        # lat1, lon1: coordinates of the first point
        # lat2, lon2: coordinates of the second point
        # Returns the azimuth angle in radians
        res = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        # x1, y1, _ = Antenna.llh_to_ecef(lat1, lon1, 0) 
        # x2, y2, _ = Antenna.llh_to_ecef(lat2, lon2, 0)
        
        # Not the true azimuth
        # return np.arctan2(lon2-lon1, lat2-lat1)

        # return np.arctan2(x2 - x1, y2 - y1)
        return np.deg2rad(res['azi2'])

    def plotPolar(self, slice='phi', value=np.pi/2):
        fig = go.Figure() 

        # Horizontal slice (theta = pi/2)
        slice_t = self.df[round(self.df[slice], 2) == round(value,2)]
        r_key = 'phi' if slice == 'theta' else 'theta'
        fig.add_trace(go.Scatterpolar(
            r = slice_t['gain_total_db'],
            theta = np.rad2deg(slice_t[r_key]),
            mode = 'lines',
            name = 'Figure 8',
            line_color = 'peru'
        ))
        
        fig.update_layout(
            title = 'Antenna Pattern at ' + slice + ' = ' + str(np.rad2deg(value)) + ' degrees',
            showlegend = False
        )
            
        fig.show()


    def plot3d(self):
        theta = np.array(self.df["theta"]);
        phi = np.array(self.df["phi"]);
        power = np.array(self.df["gain_total_db"]);
        
        phi_grid, theta_grid = np.meshgrid(np.unique(phi), np.unique(theta))

        displacement = max(abs(power.min()), abs(power.max()))
        etotal_grid = griddata((phi, theta), power, (phi_grid, theta_grid), method='linear') + displacement

        x = etotal_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y = etotal_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z = etotal_grid * np.cos(theta_grid)

        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=etotal_grid-displacement, colorscale='jet')])

        fig.update_layout(
            title='3D Antenna Radiation Pattern',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=800
        )

        fig.show() 
        
    def xyz_radiation(self):
        theta = np.array(self.df["theta"]);
        phi = np.array(self.df["phi"]);
        power = np.array(self.df["gain_total_db"]);
        
        phi_grid, theta_grid = np.meshgrid(np.unique(phi), np.unique(theta))

        displacement = max(abs(power.min()), abs(power.max()))
        etotal_grid = griddata((phi, theta), power, (phi_grid, theta_grid), method='linear') + displacement

        x = etotal_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y = etotal_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z = etotal_grid * np.cos(theta_grid)
        
        return x, y, z, etotal_grid
        
        
        
        
    def generateStl(self, fname): 
        theta = np.array(self.df["theta"]);
        phi = np.array(self.df["phi"]);
        power = np.array(self.df["gain_total_db"]);
        
        phi_grid, theta_grid = np.meshgrid(np.unique(phi), np.unique(theta))

        displacement = max(abs(power.min()), abs(power.max()))
        etotal_grid = griddata((phi, theta), power, (phi_grid, theta_grid), method='linear') + displacement

        x = etotal_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y = etotal_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z = etotal_grid * np.cos(theta_grid)

        mesh = pv.StructuredGrid(x,y,z)
        surface = mesh.extract_surface()
        triangulated = surface.triangulate()
        
        p = pv.Plotter()
        p.add_mesh(mesh)
        p.show()
        
        # pv.save_meshio(f"{fname}.stl",triangulated)
        triangulated.save(f"{fname}.stl")
        
        

    def read(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                ln  = line.split()
                if len(ln) > 5 and ln[0] != "Phi":
                    self.phi.append(float(ln[0]))
                    self.theta.append(float(ln[1]))
                    self.gain_total_db.append(float(ln[2]))
                    self.gain_phi_db.append(float(ln[3]))
                    self.gain_theta_db.append(float(ln[4]))

        self.phi = np.array(self.phi)   
        self.theta = np.array(self.theta)
        self.gain_total_db = np.array(self.gain_total_db)
        self.gain_phi_db = np.array(self.gain_phi_db)
        self.gain_theta_db = np.array(self.gain_theta_db)
        self.gain_total = 10**(self.gain_total_db/10)
        self.gain_phi = 10**(self.gain_phi_db/10)
        self.gain_theta = 10**(self.gain_theta_db/10)
        
        self.df = pd.DataFrame({'phi': self.phi, 'theta': self.theta, 'gain_total': self.gain_total, 'gain_total_db': self.gain_total_db, 'gain_phi': self.gain_phi, 'gain_phi_db': self.gain_phi_db, 'gain_theta': self.gain_theta, 'gain_theta_db': self.gain_theta_db})
        # self.df = pd.DataFrame({'phi': self.phi, 'theta': self.theta, 'gain_total': self.gain, 'gain_total_db': self.gain_db, 'gain_phi': self.gain_phi, 'gain_phi_db': self.gain_phi_db, 'gain_theta': self.gain_theta, 'gain_theta_db': self.gain_theta_db})
        # self.df = pd.DataFrame({'phi': self.phi, 'theta': self.theta, 'gain': self.gain, 'gain_db': self.gain_db})

    def getGain(self, phi, theta):
        # Interpolates the gain from the given phi and theta
        gain = griddata((self.phi, self.theta), self.gain_total_db, (phi, theta), method='nearest')
        return gain
    
    ## Predefined antenna patterns
    def readTxAntenna(self):
        _FNAME = "Antenna Measurements/RM-WB1-DN-BLK Upside Down/RM-WB1-DN-BLK Upside Down-F3500.txt" # 3.5 GHz
        self.read(_FNAME)

    def readRxAntenna(self):
        _FNAME = "Antenna Measurements/SA-1400-5900/SA-1400-5900-F3500.txt" # 3.5 GHz
        self.read(_FNAME)

