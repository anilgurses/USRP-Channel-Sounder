from dataclasses import dataclass, asdict, field
from math import pi
from arrow import get
from more_itertools import first
import numpy as np
from geopy.distance import geodesic
import time

from utils.constants import *
from utils.freq_sync import *
from utils.channel import *
from utils.vhcl_processor import *
from utils.antenna import *

TO_THRESHOLD = 0.02
CIR_OFFSET = 1000 # Offset for the CIR delay
CIR_START = 300

@dataclass
class SignalMetric:
    # Parameters
    time: np.float32 # Just in case
    center_freq: np.float32
    dist: np.float32
    h_dist: np.float32
    v_dist: np.float32
    
    wav_type: str
    
    # Calculated metrics
    detected: bool
    power: np.ndarray
    avgPower: np.float16 
    freq_offset: np.float64
    path_loss: np.ndarray
    avg_pl: np.float16
    est_dist: np.float32
    peaks: np.ndarray
    orig_peaks: np.ndarray
    start_point: np.uint32
    
    aod_theta: np.float32
    aod_phi: np.float32
    aoa_theta: np.float32 
    aoa_phi: np.float32
    
    stage: str
    
    vehicle: VehicleMetric
    
    # TODO to be replaced with Channel class
    corr: np.ndarray
    save_corr: bool = False

    # TODO Not implemented yet
    snr: np.float32 = np.float32(0.0)
    rsrp: np.float32 = np.float32(0.0)
    shadowing: int = 0
    multipath: int = 0
    delay: int = 0
    doppler: int = 0
    
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        return f"SignalMetric: time={self.time}, center_freq={self.center_freq}, dist={self.dist}, h_dist={self.h_dist}, v_dist={self.v_dist}, wav_type={self.wav_type}, detected={self.detected}, snr={self.snr}, rsrp={self.rsrp}, power={self.power}, avgPower={self.avgPower}, freq_offset={self.freq_offset}, path_loss={self.path_loss}, avg_pl={self.avg_pl}, shadowing={self.shadowing}, multipath={self.multipath}, delay={self.delay}, doppler={self.doppler}, est_dist={self.est_dist}, peaks={self.peaks}, vehicle={self.vehicle}, corr={self.corr}, save_corr={self.save_corr}"
    
    def __repr__(self) -> str:
        return f"SignalMetric: time={self.time}, center_freq={self.center_freq}, dist={self.dist}, h_dist={self.h_dist}, v_dist={self.v_dist}, wav_type={self.wav_type}, detected={self.detected}, snr={self.snr}, rsrp={self.rsrp}, power={self.power}, avgPower={self.avgPower}, freq_offset={self.freq_offset}, path_loss={self.path_loss}, avg_pl={self.avg_pl}, shadowing={self.shadowing}, multipath={self.multipath}, delay={self.delay}, doppler={self.doppler}, est_dist={self.est_dist}, peaks={self.peaks}, vehicle={self.vehicle}, corr={self.corr}, save_corr={self.save_corr}"

    # Not needed anymore
    def __to_dict__(self):
        dct = asdict(self) 
        v_dct = self.vehicle.__to_dict__()
        del v_dct["time"]
        del dct["vehicle"]
        dct.update(v_dct)
        return dct

class SigProcessor:
    """_summary_
        Received signal processor
    """
    def __init__(self, config, wav1, wav2, total_len) -> None:
        # Parameters that are required for calculation
        self.config = config 
        self.ref_signal = wav1
        self.ofdm_signal = wav2
        ## TODO: replace with frame_len
        self.total_len = total_len 
        self.start_point = np.inf
    
    def getIndex(self, cir):
        return np.argmax(cir)
    
    def getPeaks(self, cir, prm=40):
        peaks, _ = signal.find_peaks(np.abs(cir), distance=self.config.WAV_OPTS.SEQ_LEN, prominence=prm)
        return peaks 

    ## dBFs to dBm
    def calcPowerdBm(self, sig_of_int):
        return 20 * np.log10(np.var(sig_of_int) + 1e-13) 
    
    def calcPathLoss(self, sig_of_int, tx_ref_dbm, rx_ref_dbm):
        return tx_ref_dbm + self.calcPowerdBm(self.ref_signal) - rx_ref_dbm - self.calcPowerdBm(sig_of_int)
    
    def calcDist(self, ind):
        """Calculate the distance between the receiver and the transmitter"""
        est_dist = (ind - self.config.WAV_OPTS.SEQ_LEN - N_DELAY_SAMPLE_USRP) * (1/self.config.USRP_CONF.SAMPLE_RATE) * SPEED_OF_LIGHT
        return est_dist
    
    def crop_signal(self, rcv, start, end):
        """Crop the received signal to the signal of interest"""
        return rcv[start:end]

    def getFirstPeak(self, rcv, first_peak, last_peak):
        end = last_peak + self.config.WAV_OPTS.SEQ_LEN
        self.sig_of_int = rcv[first_peak:end]

        return self.sig_of_int
    
    def moose_alg(self, preamble, fs):
        num_samples = preamble.size
        self_ref_size = num_samples // 2
        first_half = np.vstack(preamble[:self_ref_size])
        second_half = np.vstack(preamble[self_ref_size:])
        phase_offset,_,_,_ = np.linalg.lstsq(first_half, second_half, rcond=None)
        # use phase offset to find frequency
        freq_shift = np.angle(phase_offset)/(2*np.pi)/(1/fs*self_ref_size) 
        freq_shift = np.squeeze(np.array(freq_shift))
        return freq_shift
    
    def correctFreq(self, rcv, preamble):
        """Correct the frequency offset in the received signal"""
        freq_shift = self.moose_alg(preamble, self.config.USRP_CONF.SAMPLE_RATE)
        Ts = 1/self.config.USRP_CONF.SAMPLE_RATE
        t = np.arange(0, Ts*len(rcv) - Ts, Ts) 
        return freq_shift.flatten(), rcv * np.exp(-1j*2*np.pi*freq_shift*t)

    def getPreamble(self, rcv, peaks):
        preamble = rcv[peaks[0]:peaks[0]+self.config.WAV_OPTS.SEQ_LEN*2]
        return preamble
    
    def getCIR(self, rcv, ref, normalize=False):
        '''
        Calculates the channel impulse for received signal rcv 
        '''
        xcorr = signal.correlate(rcv, ref, mode="full", method="fft")
        lags = signal.correlation_lags(len(ref), len(rcv))
        # xcorr = np.abs(xcorr) 

        if normalize:
            xcorr /= np.max(xcorr) 

        return xcorr, lags
    
    def getStartPoint(self):
        return self.start_point
    
    def zeroMetric(self,vehicle):
        sgnlMetric = SignalMetric() 
        sgnlMetric.time = 0.0
        sgnlMetric.center_freq = 0.0
        sgnlMetric.dist = 0.0
        sgnlMetric.h_dist = 0.0
        sgnlMetric.v_dist = 0.0
        sgnlMetric.wav_type = ""
        sgnlMetric.detected = False
        sgnlMetric.power = np.array([])
        sgnlMetric.avgPower = 0.0
        sgnlMetric.freq_offset = 0.0
        sgnlMetric.path_loss = np.array([])
        sgnlMetric.avg_pl = 0.0
        sgnlMetric.est_dist = 0.0
        sgnlMetric.peaks = np.array([])
        sgnlMetric.start_point = 0
        sgnlMetric.aod_theta = 0.0
        sgnlMetric.aod_phi = 0.0
        sgnlMetric.aoa_theta = 0.0
        sgnlMetric.aoa_phi = 0.0
        sgnlMetric.stage = ""
        sgnlMetric.vehicle = vehicle
        sgnlMetric.corr = np.array([])
        sgnlMetric.save_corr = False
        sgnlMetric.snr = 0.0
        sgnlMetric.rsrp = 0.0
        sgnlMetric.shadowing = 0
        sgnlMetric.multipath = 0
        sgnlMetric.delay = 0
        sgnlMetric.doppler = 0
        return sgnlMetric

    def process(self, r_time, rcv, vehicle_metric, save_corr=True):
        """Process the received signal step by step"""
        metrics = SignalMetric()
        
        metrics.time = np.float32(r_time)
        metrics.center_freq = self.config.USRP_CONF.CENTER_FREQ / 1e6 # MHz
        metrics.wav_type = self.config.WAVEFORM
        
        # 1 - Detect the signal 
        xcorr, _ = self.getCIR(rcv, self.ref_signal)
        peaks = self.getPeaks(xcorr)
       
        tmp_xcorr = xcorr
        if len(peaks) == 0:
            ## Making sure that signal exists
            return self.zeroMetric(vehicle_metric)
        
        preamble = self.getPreamble(rcv, peaks)
        freq_shift, rcv = self.correctFreq(rcv, preamble)
        xcorr, _ = self.getCIR(rcv, self.ref_signal)
        peaks = self.getPeaks(xcorr)
        # For future reference
         
        if len(peaks) == 0:
            return self.zeroMetric(vehicle_metric)
        
        first_peak = peaks[0]
        metrics.start_point = first_peak
        metrics.detected = True
        
        metrics.freq_offset = freq_shift[0]
        metrics.orig_peaks = peaks

        _crop_num_samples = peaks[0] if CIR_OFFSET > peaks[0] else CIR_OFFSET
        _crop_start = peaks[0] - _crop_num_samples 
        _crop_end = peaks[0] + self.total_len + CIR_OFFSET // 3
        
        rcv = self.crop_signal(rcv, _crop_start, _crop_end)
        
        # This might be too much
        xcorr = xcorr[CIR_START:peaks[0]+self.total_len]
        peaks = self.getPeaks(xcorr)
        
        metrics.peaks = peaks
        
        if save_corr:
            metrics.corr = tmp_xcorr
            metrics.save_corr = True
        
        # 2 - Calculate the power of the signal
        _pr_len = len(self.ref_signal)
        
        metrics.power = np.array([self.calcPowerdBm(rcv[peak:peak+_pr_len]) for peak in peaks])
            
        if metrics.power.any(): 
            metrics.avgPower = np.mean(metrics.power)
        
        if getattr(metrics, 'avgPower', None) is None:
            metrics.avgPower = np.nan
        
        # 3 - Calculate path loss 
        # TODO forgot to change it on the config before transmission
        self.config.TX_REF_DBM = 19.97 # dBm
        self.config.RX_REF_DBM = -50.68 # dBm
        
        metrics.path_loss = np.array([self.calcPathLoss(rcv[peak:peak+_pr_len], self.config.TX_REF_DBM, self.config.RX_REF_DBM) for peak in peaks])
        if metrics.path_loss.any():
            metrics.avg_pl = np.mean(metrics.path_loss)
            
        if getattr(metrics, 'avg_pl', None) is None:
            metrics.avg_pl = np.nan
        
        # 4 - Calculate the distance
        metrics.est_dist = self.calcDist(first_peak)
        
        # 5 - Vehicle related metrics
        metrics.vehicle = vehicle_metric
        metrics.h_dist = np.float32(geodesic(LW1, (vehicle_metric.lat, vehicle_metric.lon)).meters)
        metrics.v_dist = np.abs(vehicle_metric.alt - H_TOWER_LW1)
        metrics.dist = np.sqrt(metrics.h_dist**2 + metrics.v_dist**2)
        
        if vehicle_metric.vel_z < TO_THRESHOLD and vehicle_metric.vel_z > -TO_THRESHOLD:
            metrics.stage = "Flight"
        elif vehicle_metric.vel_z > TO_THRESHOLD:
            metrics.stage = "Takeoff"
        else:
            metrics.stage = "Landing"
        
        # Update the tower parameter with configuration
        metrics.aod_phi = Antenna.get_elevation_angle(vehicle_metric.lat, vehicle_metric.lon, LW1[0], LW1[1], vehicle_metric.alt, H_TOWER_LW1)
        metrics.aod_theta = Antenna.get_azimuth_angle(vehicle_metric.lat, vehicle_metric.lon, LW1[0], LW1[1])
        
        ## It's mirror of aod
        metrics.aoa_phi = Antenna.get_elevation_angle(LW1[0], LW1[1], vehicle_metric.lat, vehicle_metric.lon, H_TOWER_LW1, vehicle_metric.alt)
        metrics.aoa_theta = Antenna.get_azimuth_angle(LW1[0], LW1[1], vehicle_metric.lat, vehicle_metric.lon)
               
               
        # 6 - Channel related metrics
        
        return metrics
    
    # Shift the CIR by the offset value 
    # then crop the CIR to the size of the reference signal
    def adjustCIR(self, cir, ref, first_peak):
        cir = np.roll(cir, first_peak-ref)
        cir = cir[:self.total_len]
        return cir