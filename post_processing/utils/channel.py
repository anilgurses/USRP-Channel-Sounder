import numpy as np
from scipy import signal
from dataclasses import dataclass


@dataclass
class ChannelMetric:
    time: int
    center_freq: int
    bw: int
    sig_of_int: np.ndarray
    cir: np.ndarray
    pdp: np.ndarray
    reg_ref: np.ndarray
    reflects: np.ndarray
    n_rflcts: int
    
    free_path_loss: float
    
    
## It will have my algorithm to generate channel impulse response and other details such as path loss
class Channel:
    def __init__(self) -> None:
        pass
    
    def generateCIR(self, config, tx, rx):
        pass
    
    def pathLoss(self, freq, dist):
        pass
    
    def calculateShadowing(self, config):   
        pass
 
class RMa3GPPChannelModel(Channel):
    def __init__(self) -> None:
        pass
    
    def generateCIR(self, config, tx, rx):
        pass
    
    @staticmethod
    def pathLoss(d_2d, d_3D, fc, h_bs = 10, h_ue = 1.5, h = 5, w = 10):
        PL_RMa_LOS, PL_RMa_NLOS = 0.0, 0.0
        
        if h_ue < 1 : 
            return np.nan
        d_BP = 2*np.pi*h_bs*h_ue*fc 
        if d_BP == 0.0 or d_3D == 0.0:
            return np.nan
        
        PL_1 = 20*np.log10(40*np.pi*d_3D*fc/3) +  min(0.03*h**1.72 , 10)*np.log10(d_3D) - min(0.044*h**1.72 , 14.77) + 0.002*np.log10(h) *d_3D
        PL_2 = PL_1 + 40*np.log10(d_3D/d_BP)
        if  10 < d_2d and d_2d <= d_BP:
            PL_RMa_LOS = PL_1 + 4 * np.random.normal(0, 1) # Shadow fading
        elif d_2d < 10:
            PL_RMa_Los = np.nan
        elif d_BP < d_2d and d_2d <= 10*1000:
            PL_RMa_LOS = PL_2 + 6 * np.random.normal(0, 1)
            
        PL_RMa_NLOS_2 = 161.04 - 7.1*np.log10(40*w) + 7.5*np.log10(h) -(24.37 - 3.7*(h/h_bs)*(h/h_bs))*np.log10(h_bs) + (43.42 - 3.1 * np.log10(h_bs))*(np.log10(d_3D)-3) + 20*np.log10(fc) - (3.2*np.log10(11.75*h_ue)**2 - 4.97)
        
        if 10 < d_2d and d_2d <= 5*1000:
            PL_RMa_NLOS = max(PL_RMa_LOS, PL_RMa_NLOS_2) + 8
        
        return PL_RMa_LOS
    
    def calculateShadowing(self, config):   
        pass


# #Pathloss [dB], fc is in GHz and d is in meters
class UMa3GPPChannelModel(Channel):
    def __init__(self) -> None:
        super().__init__()
        pass 
    
    def generateCIR(self, config, tx, rx):
        pass
    
    def pathLoss(self, d_2d, fc, h_bs = 10, h_ue = 1.5, h = 2, w = 20):
        PL_UMa_LOS, PL_UMa_NLOS = 0.0, 0.0
        
        h_E = 1.0
        g_d2D = 0
        if g_d2D <= 18:
            g_d2D = 0
        if g_d2D <= d_2D:
            g_d2D = 1.25*np.power((d_2D/100),3)*np.exp(-d_2D/150)
        
        if h_ue < 13:
            C_d2d_and_hUT = 0
        if 13<=h_ue and h_ue<=23:
            C_d2d_and_hUT = np.power(((h_ue-13)/10),1.5)*g_d2D
            
        probability = 1/(1 + C_d2d_and_hUT)
        
        p = np.random.uniform()
        if p < probability:
            h_E = 1.0
        else:
            h_E = h_ue - 1.5
            
        h_BS_2 = h_bs - h_E
        h_UT_2 = h_ue - h_E
        d_BP_2 = 2*np.pi*h_BS_2*h_UT_2*fc
        d_3D = np.sqrt(d_2d**2 + abs(h_bs-h_ue)**2)
        
        PL_1 = 28.0 + 22*np.log10(d_3D) + 20*np.log10(fc)
        PL_2 = 28.0 + 40*np.log10(d_3D) + 20*np.log10(fc) - 9*np.log10(np.power(d_BP_2,2) + np.power((h_bs - h_ue),2))
        
        if  10 < d_2d and d_2d <= d_BP_2:
            PL_UMa_LOS = PL_1 + 4
        elif d_BP_2 < d_2d and d_2d <= 5*1000:
            PL_UMa_LOS = PL_2 + 4
        
        PL_UMa_NLOS_2 = 13.54 + 39.08*np.log10(d_3D) + 20*np.log10(fc) - 0.6*(h_ue - 1.5)
        if 10 < d_2d and d_2d <= 5*1000:
            PL_UMa_NLOS = max(PL_UMa_LOS, PL_UMa_NLOS_2) + 6
            
        return PL_UMa_LOS , PL_UMa_NLOS
        
    
    def calculateShadowing(self, config):
        pass
    
    
class ITUChannelModel(Channel):
    def __init__(self) -> None:
        pass
    
    def generateCIR(self, config, tx, rx):
        pass
    
    def pathLoss(self, freq, dist):
        pass
    
    def calculateShadowing(self, config):   
        pass
    
    
class AERPAWChannelModel(Channel):
    def __init__(self) -> None:
        pass
    
    def generateCIR(self, config, tx, rx):
        pass
    
    def pathLoss(self, freq, dist):
        pass
    
    def calculateShadowing(self, config):   
        pass