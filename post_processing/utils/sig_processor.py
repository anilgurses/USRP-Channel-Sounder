from dataclasses import dataclass, asdict, field
import numpy as np
from geopy.distance import geodesic
import time
from scipy import signal
import csv
import os

from utils.constants import *
from utils.channel import *
from utils.vhcl_processor import *
from utils.antenna import *

TO_THRESHOLD = 0.02
CIR_OFFSET = 1000 # Offset for the CIR delay
CIR_START = 300

# Link-budget constants
TX_REF_DBM = 15.0       # B210 measured TX output power at antenna port (dBm)
CABLE_LOSS_DB = 2.0      # Estimated total TX + RX cable/connector loss (dB)

# Calibration CSV: maps USRP serial → RX reference power at each gain setting.
# Columns are gain values; entries are the RF input power (dBm) that produces
# the reference digital level at that gain.
_POWER_REFS_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'power_refs.csv')

def _load_rx_ref(gain: float, serial: str = '', csv_path: str = _POWER_REFS_CSV) -> float:
    if not os.path.isfile(csv_path):
        # Fallback: average of both serials at gain=70 (pre-computed)
        return -48.83

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Column headers after serial are gain values
        gain_cols = [float(g.strip()) for g in header[1:]]
        rows = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            ser = row[0].strip()
            vals = [float(v.strip()) for v in row[1:]]
            rows[ser] = vals

    if not rows:
        return -48.83

    # Pick the right serial or average
    if serial and serial in rows:
        vals = rows[serial]
    else:
        vals = np.mean(list(rows.values()), axis=0).tolist()

    return float(np.interp(gain, gain_cols, vals))

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
    snr: np.ndarray
    avgSnr: np.float32
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
    rsrp: np.float32 = np.float32(0.0)
    shadowing: int = 0
    multipath: int = 0
    delay: int = 0
    doppler_shift: np.float32 = 0.0
    doppler_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_delay_spread: np.float32 = 0.0
    k_factor: np.float32 = 0.0
    
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        return f"SignalMetric: time={self.time}, center_freq={self.center_freq}, dist={self.dist}, h_dist={self.h_dist}, v_dist={self.v_dist}, wav_type={self.wav_type}, detected={self.detected}, snr={self.snr}, rsrp={self.rsrp}, power={self.power}, avgPower={self.avgPower}, freq_offset={self.freq_offset}, path_loss={self.path_loss}, avg_pl={self.avg_pl}, shadowing={self.shadowing}, multipath={self.multipath}, delay={self.delay}, doppler_shift={self.doppler_shift}, est_dist={self.est_dist}, peaks={self.peaks}, vehicle={self.vehicle}, corr={self.corr}, save_corr={self.save_corr}"
    
    def __repr__(self) -> str:
        return f"SignalMetric: time={self.time}, center_freq={self.center_freq}, dist={self.dist}, h_dist={self.h_dist}, v_dist={self.v_dist}, wav_type={self.wav_type}, detected={self.detected}, snr={self.snr}, rsrp={self.rsrp}, power={self.power}, avgPower={self.avgPower}, freq_offset={self.freq_offset}, path_loss={self.path_loss}, avg_pl={self.avg_pl}, shadowing={self.shadowing}, multipath={self.multipath}, delay={self.delay}, doppler_shift={self.doppler_shift}, est_dist={self.est_dist}, peaks={self.peaks}, vehicle={self.vehicle}, corr={self.corr}, save_corr={self.save_corr}"

    # Not needed anymore
    def __to_dict__(self):
        dct = asdict(self)
        v_dct = self.vehicle.__to_dict__()
        del v_dct["time"]
        del dct["vehicle"]
        dct.update(v_dct)
        return dct

    def to_scalar_dict(self):
        """Return only scalar values — lightweight dict for DataFrame construction.

        Excludes large numpy arrays (corr, power, snr, path_loss, peaks, etc.)
        that are not persisted to CSV, drastically reducing memory when results
        are collected from multiprocessing workers.
        """
        d = {
            'time': self.time, 'center_freq': self.center_freq,
            'dist': self.dist, 'h_dist': self.h_dist, 'v_dist': self.v_dist,
            'wav_type': self.wav_type, 'detected': self.detected,
            'avgPower': self.avgPower, 'avgSnr': self.avgSnr,
            'freq_offset': self.freq_offset, 'avg_pl': self.avg_pl,
            'est_dist': self.est_dist, 'start_point': self.start_point,
            'aod_theta': self.aod_theta, 'aod_phi': self.aod_phi,
            'aoa_theta': self.aoa_theta, 'aoa_phi': self.aoa_phi,
            'stage': self.stage,
            'rsrp': self.rsrp, 'shadowing': self.shadowing,
            'multipath': self.multipath, 'delay': self.delay,
            'doppler_shift': self.doppler_shift,
            'rms_delay_spread': self.rms_delay_spread,
            'k_factor': self.k_factor,
        }
        v = self.vehicle
        d.update({
            'lat': v.lat, 'lon': v.lon, 'alt': v.alt,
            'pitch': v.pitch, 'yaw': v.yaw, 'roll': v.roll,
            'vel_x': v.vel_x, 'vel_y': v.vel_y, 'vel_z': v.vel_z,
            'heading': v.heading, 'speed': v.speed,
        })
        return d

class SigProcessor:
    def __init__(self, config, wav1, wav2, total_len, interpolate_rate=1) -> None:
        # Parameters that are required for calculation
        self.config = config 
        self.ref_signal = wav1
        self.ofdm_signal = wav2
        ## TODO: replace with frame_len
        self.total_len = total_len 
        self.start_point = np.inf
        self.interpolate_rate = interpolate_rate
    
    def getPeaks(self, cir, prm=40):
        peaks, _ = signal.find_peaks(np.abs(cir), distance=self.config.WAV_OPTS.SEQ_LEN, prominence=prm)
        return peaks 

    def corrIndexToSampleIndex(self, peak_idx):
        """Map a full-correlation peak index back to the receive-sample start."""
        return int(peak_idx) - (len(self.ref_signal) - 1)

    def corrPeaksToSampleIndices(self, peaks, signal_len):
        """Convert correlation indices to valid receive-sample indices."""
        if len(peaks) == 0:
            return np.array([], dtype=np.int64)
        starts = np.asarray(peaks, dtype=np.int64) - (len(self.ref_signal) - 1)
        valid = (starts >= 0) & (starts + len(self.ref_signal) <= signal_len)
        return starts[valid]

    def selectFramePeaks(self, peak_samples):
        """Keep the peaks that belong to the first detected burst only."""
        if len(peak_samples) == 0:
            return np.array([], dtype=np.int64)
        first_peak = int(peak_samples[0])
        in_frame = (peak_samples >= first_peak) & (peak_samples < first_peak + self.total_len)
        return np.asarray(peak_samples[in_frame], dtype=np.int64)

    ## Digital power (dBFS) — var() returns mean power E[|x|²], so use 10·log10
    def calcPowerdBm(self, sig_of_int):
        return 10 * np.log10(np.var(sig_of_int) + 1e-13)
    
    def calcPathLoss(self, sig_of_int, tx_ref_dbm, rx_ref_dbm):
        return tx_ref_dbm + self.calcPowerdBm(self.ref_signal) - rx_ref_dbm - self.calcPowerdBm(sig_of_int)
    
    def calcSNR(self, sig_of_interest, seq):
        if len(sig_of_interest) < len(seq):
            raise ValueError("The length of the signal is less than the length of the reference sequence")
        correlation = np.correlate(sig_of_interest, seq, mode='full')
        peak_index = np.argmax(np.abs(correlation))
        signal_est = correlation[peak_index] / len(seq)
        
        signal_power = np.abs(signal_est)**2

        reconstructed_signal = signal_est * seq
        residual_noise = sig_of_interest[:len(reconstructed_signal)] - reconstructed_signal
        noise_power = np.mean(np.abs(residual_noise)**2)
        snr_estimated = signal_power / noise_power
        return 10 * np.log10(snr_estimated)
    
    def calcDist(self, ind):
        est_dist = (ind - N_DELAY_SAMPLE_USRP) * (1/self.config.USRP_CONF.SAMPLE_RATE) * SPEED_OF_LIGHT
        return est_dist
    
    def crop_signal(self, rcv, start, end):
        return rcv[start:end]

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
    
    def correctFreq(self, rcv, freq_shift):
        Ts = 1/self.config.USRP_CONF.SAMPLE_RATE
        # Might need to add - Ts
        t = np.arange(len(rcv)) * Ts 
        return rcv * np.exp(-1j*2*np.pi*freq_shift*t)

    def calcWidebandPSD(self, rcv):
        fs = self.config.USRP_CONF.SAMPLE_RATE
        f, Pxx = signal.welch(rcv, fs, nperseg=1024, return_onesided=False)
        return f, Pxx

    def _threshold_pdp(self, power_pdp, noise_floor, noise_margin_db=3):
        # Set threshold a few dB above the provided noise floor
        threshold = noise_floor * 10**(noise_margin_db / 10)

        # Any power level below the threshold is noise
        power_pdp[power_pdp < threshold] = 0
        return power_pdp

    def calcRMSDelaySpread(self, cir, sample_rate, noise_floor, noise_margin_db=3):
        power_pdp = np.abs(cir)**2
        power_pdp = self._threshold_pdp(power_pdp, noise_floor, noise_margin_db)
        
        # Find first and last path
        paths = np.where(power_pdp > 0)[0]
        if len(paths) == 0:
            return 0.0
        first_path_idx = paths[0]
        last_path_idx = paths[-1]
        
        # Crop PDP to only include paths
        power_pdp_cropped = power_pdp[first_path_idx:last_path_idx+1]
        
        total_power = np.sum(power_pdp_cropped)
        if total_power == 0:
            return 0.0
            
        time_delays = np.arange(len(power_pdp_cropped)) / sample_rate
        
        mean_delay = np.sum(time_delays * power_pdp_cropped) / total_power
        
        rms_delay_spread = np.sqrt(np.sum(((time_delays - mean_delay)**2) * power_pdp_cropped) / total_power)
        
        return rms_delay_spread

    def calcKFactor(self, cir, noise_floor, noise_margin_db=3):
        power_pdp = np.abs(cir)**2
        power_pdp = self._threshold_pdp(power_pdp, noise_floor, noise_margin_db)

        if not power_pdp.any():
            return 0.0
        
        los_peak_idx = np.argmax(power_pdp)
        los_power = power_pdp[los_peak_idx]
        
        nlos_power = np.sum(power_pdp) - los_power
        
        if nlos_power == 0:
            return 40.0  # Cap at 40 dB — pure LoS (no measurable NLOS power)
            
        k_factor = los_power / nlos_power
        return 10 * np.log10(k_factor)

    def getPreamble(self, rcv, peak_samples):
        preamble = rcv[peak_samples[0]:peak_samples[0]+self.config.WAV_OPTS.SEQ_LEN*2]
        return preamble
    
    def getCIR(self, rcv, ref, normalize=False):
        xcorr = signal.correlate(rcv, ref, mode="full", method="fft")
        lags = signal.correlation_lags(len(rcv), len(ref), mode="full")
        # xcorr = np.abs(xcorr) 

        if normalize:
            xcorr /= np.max(xcorr) 

        return xcorr, lags
    
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
        sgnlMetric.snr = np.array([])
        sgnlMetric.avgSnr = 0.0
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
        sgnlMetric.rsrp = 0.0
        sgnlMetric.shadowing = 0
        sgnlMetric.multipath = 0
        sgnlMetric.delay = 0
        sgnlMetric.doppler_shift = 0.0
        sgnlMetric.doppler_spectrum = np.array([])
        sgnlMetric.rms_delay_spread = 0.0
        sgnlMetric.k_factor = 0.0
        return sgnlMetric
    
    def process(self, r_time, rcv, vehicle_metric, tx_vehicle_metric=None, save_corr=False):
        """Process the received signal step by step"""
        metrics = SignalMetric()
        
        metrics.time = np.float32(r_time)
        metrics.center_freq = self.config.USRP_CONF.CENTER_FREQ / 1e6 # MHz
        metrics.wav_type = self.config.WAVEFORM
        
        # 1 - Detect the signal 
        xcorr, _ = self.getCIR(rcv, self.ref_signal)
        peaks = self.getPeaks(xcorr)
        peak_samples = self.corrPeaksToSampleIndices(peaks, len(rcv))
        peak_samples = peak_samples[
            peak_samples + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv)
        ]
       
        if save_corr:
            tmp_xcorr = xcorr
        if len(peak_samples) == 0:
            ## Making sure that signal exists
            return self.zeroMetric(vehicle_metric)
        
        # Coarse frequency estimation
        preamble_coarse = self.getPreamble(rcv, peak_samples)
        freq_shift_coarse = self.moose_alg(preamble_coarse, self.config.USRP_CONF.SAMPLE_RATE)
        rcv_coarse_corrected = self.correctFreq(rcv, freq_shift_coarse)

        # Fine frequency estimation
        xcorr_fine, _ = self.getCIR(rcv_coarse_corrected, self.ref_signal)
        peaks_fine = self.getPeaks(xcorr_fine)
        peak_samples_fine = self.corrPeaksToSampleIndices(peaks_fine, len(rcv_coarse_corrected))
        peak_samples_fine = peak_samples_fine[
            peak_samples_fine + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv_coarse_corrected)
        ]

        if len(peak_samples_fine) == 0:
            # Fine step degraded detection; keep coarse correction but use the
            # xcorr from the coarse-corrected signal and re-detect peaks.
            freq_shift = freq_shift_coarse
            rcv = rcv_coarse_corrected
            xcorr = xcorr_fine
            peaks = self.getPeaks(xcorr_fine, prm=20)  # lower prominence threshold
            peak_samples = self.corrPeaksToSampleIndices(peaks, len(rcv))
            peak_samples = peak_samples[
                peak_samples + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv)
            ]
        else:
            preamble_fine = self.getPreamble(rcv_coarse_corrected, peak_samples_fine)
            freq_shift_fine = self.moose_alg(preamble_fine, self.config.USRP_CONF.SAMPLE_RATE)
            
            freq_shift = freq_shift_coarse + freq_shift_fine
            
            # Correct original rcv with total frequency shift
            rcv = self.correctFreq(rcv, freq_shift)
            
            # Recalculate xcorr and peaks with the fully corrected rcv
            xcorr, _ = self.getCIR(rcv, self.ref_signal)
            peaks = self.getPeaks(xcorr)
            peak_samples = self.corrPeaksToSampleIndices(peaks, len(rcv))
            peak_samples = peak_samples[
                peak_samples + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv)
            ]

        # For future reference

        if len(peak_samples) == 0:
            return self.zeroMetric(vehicle_metric)
        peak_samples = self.selectFramePeaks(peak_samples)
        if len(peak_samples) == 0:
            return self.zeroMetric(vehicle_metric)
        
        # Experimental
        # --- Interpolation Logic ---
        cir_for_delay_spread = xcorr
        sample_rate_for_delay_spread = self.config.USRP_CONF.SAMPLE_RATE

        if self.interpolate_rate > 1:
            interp_rcv = signal.resample(rcv, len(rcv) * self.interpolate_rate)
            interp_ref = signal.resample(self.ref_signal, len(self.ref_signal) * self.interpolate_rate)
            
            cir_for_delay_spread, _ = self.getCIR(interp_rcv, interp_ref)
            sample_rate_for_delay_spread = self.config.USRP_CONF.SAMPLE_RATE * self.interpolate_rate
            del interp_rcv, interp_ref
        # --- End Interpolation Logic ---

        first_peak = int(peak_samples[0])
        metrics.start_point = first_peak
        metrics.detected = True
        
        metrics.freq_offset = freq_shift
        metrics.orig_peaks = peak_samples.copy()
        metrics.peaks = peak_samples.copy()
        
        if save_corr:
            metrics.corr = tmp_xcorr
            metrics.save_corr = True
        else:
            metrics.corr = np.array([])
            metrics.save_corr = False
        
        # 2 - Calculate the power of the signal
        _pr_len = len(self.ref_signal)
        
        peak_samples = [int(p) for p in peak_samples if p + _pr_len <= len(rcv)]

        metrics.power = np.array([self.calcPowerdBm(rcv[peak:peak+_pr_len]) for peak in peak_samples])
            
        if metrics.power.any(): 
            metrics.avgPower = np.mean(metrics.power)
        
        if getattr(metrics, 'avgPower', None) is None:
            metrics.avgPower = np.nan
        
        # 3 - Calculate path loss
        # Load RX calibration from power_refs.csv for the actual gain setting
        gain = getattr(self.config.USRP_CONF, 'GAIN', 70)
        serial = getattr(self.config.USRP_CONF, 'SERIAL', '')
        self.config.TX_REF_DBM = TX_REF_DBM
        self.config.RX_REF_DBM = _load_rx_ref(gain, serial)

        metrics.path_loss = np.array([self.calcPathLoss(rcv[peak:peak+_pr_len], self.config.TX_REF_DBM, self.config.RX_REF_DBM) for peak in peak_samples])
        if metrics.path_loss.any():
            metrics.avg_pl = np.mean(metrics.path_loss)
            
        if getattr(metrics, 'avg_pl', None) is None:
            metrics.avg_pl = np.nan
        
        # 4 - Calculate the distance
        metrics.est_dist = self.calcDist(first_peak)
        
        # 5 - Vehicle related metrics
        metrics.vehicle = vehicle_metric
        if tx_vehicle_metric:
            tx_loc = (tx_vehicle_metric.lat, tx_vehicle_metric.lon)
            rx_loc = (vehicle_metric.lat, vehicle_metric.lon)
            metrics.h_dist = np.float32(geodesic(tx_loc, rx_loc).meters)
            metrics.v_dist = np.abs(vehicle_metric.alt - tx_vehicle_metric.alt)
            metrics.dist = np.sqrt(metrics.h_dist**2 + metrics.v_dist**2)
            if metrics.est_dist < metrics.dist - 50 or metrics.est_dist > metrics.dist + 50:
                return self.zeroMetric(vehicle_metric)
            metrics.aod_phi = Antenna.get_elevation_angle(vehicle_metric.lat, vehicle_metric.lon, tx_vehicle_metric.lat, tx_vehicle_metric.lon, vehicle_metric.alt, tx_vehicle_metric.alt)
            metrics.aod_theta = Antenna.get_azimuth_angle(vehicle_metric.lat, vehicle_metric.lon, tx_vehicle_metric.lat, tx_vehicle_metric.lon)
            metrics.aoa_phi = Antenna.get_elevation_angle(tx_vehicle_metric.lat, tx_vehicle_metric.lon, vehicle_metric.lat, vehicle_metric.lon, tx_vehicle_metric.alt, vehicle_metric.alt)
            metrics.aoa_theta = Antenna.get_azimuth_angle(tx_vehicle_metric.lat, tx_vehicle_metric.lon, vehicle_metric.lat, vehicle_metric.lon)
        else:
            metrics.h_dist = np.float32(geodesic(LW1, (vehicle_metric.lat, vehicle_metric.lon)).meters)
            metrics.v_dist = np.abs(vehicle_metric.alt - H_TOWER_LW1)
            metrics.dist = np.sqrt(metrics.h_dist**2 + metrics.v_dist**2)
            metrics.aod_phi = Antenna.get_elevation_angle(vehicle_metric.lat, vehicle_metric.lon, LW1[0], LW1[1], vehicle_metric.alt, H_TOWER_LW1)
            metrics.aod_theta = Antenna.get_azimuth_angle(vehicle_metric.lat, vehicle_metric.lon, LW1[0], LW1[1])
            metrics.aoa_phi = Antenna.get_elevation_angle(LW1[0], LW1[1], vehicle_metric.lat, vehicle_metric.lon, H_TOWER_LW1, vehicle_metric.alt)
            metrics.aoa_theta = Antenna.get_azimuth_angle(LW1[0], LW1[1], vehicle_metric.lat, vehicle_metric.lon)

        if abs(metrics.est_dist) < 1: # Check if est_dist is close to 0
            return self.zeroMetric(vehicle_metric)

        if vehicle_metric.vel_z < TO_THRESHOLD and vehicle_metric.vel_z > -TO_THRESHOLD:
            metrics.stage = "Flight"
        elif vehicle_metric.vel_z > TO_THRESHOLD:
            metrics.stage = "Takeoff"
        else:
            metrics.stage = "Landing"
        
        # 6 - Channel related metrics
        metrics.snr = np.array([self.calcSNR(rcv[peak:peak+_pr_len], self.ref_signal) for peak in peak_samples])
        metrics.avgSnr = np.mean(metrics.snr)

        # 7 - Wideband PSD of this frame (stored for per-frame inspection;
        #     true slow-time Doppler spectrum is computed in post-processing)
        f, Pxx = self.calcWidebandPSD(rcv)
        metrics.doppler_shift = metrics.freq_offset
        metrics.doppler_spectrum = Pxx

        # 8 - RMS Delay Spread and K-Factor
        # Find the main peak in the CIR used for delay spread
        max_peak_index = np.argmax(np.abs(cir_for_delay_spread))

        # Define the window size, scaled by interpolation rate
        # Use a smaller window (~2 μs) appropriate for typical delay spreads
        window_size = 100 * self.interpolate_rate
        half_window = window_size // 2

        start_crop = max(0, max_peak_index - half_window)
        end_crop = min(len(cir_for_delay_spread), max_peak_index + half_window)

        # Crop the CIR to focus on the channel response of a single transmission
        cropped_cir = cir_for_delay_spread[start_crop:end_crop]

        # Estimate noise floor from AFTER the signal window (not from edge effects at the start)
        noise_start = end_crop + 100 * self.interpolate_rate
        noise_end = noise_start + 200 * self.interpolate_rate
        if noise_end < len(cir_for_delay_spread):
            noise_floor_pdp = np.mean(np.abs(cir_for_delay_spread[noise_start:noise_end])**2)
        else:
            # Fallback: use region before the peak if not enough samples after
            noise_end_alt = max(0, start_crop - 100 * self.interpolate_rate)
            noise_start_alt = max(0, noise_end_alt - 200 * self.interpolate_rate)
            if noise_start_alt < noise_end_alt:
                noise_floor_pdp = np.mean(np.abs(cir_for_delay_spread[noise_start_alt:noise_end_alt])**2)
            else:
                noise_floor_pdp = np.percentile(np.abs(cropped_cir)**2, 10)

        metrics.rms_delay_spread = self.calcRMSDelaySpread(cropped_cir, sample_rate_for_delay_spread, noise_floor=noise_floor_pdp)
        metrics.k_factor = self.calcKFactor(cropped_cir, noise_floor=noise_floor_pdp)

        return metrics
