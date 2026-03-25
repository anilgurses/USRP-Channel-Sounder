from dataclasses import dataclass, asdict, field
import numpy as np
from geopy.distance import geodesic
import time
from scipy import signal

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

class SigProcessor:
    """Received signal processor"""
    def __init__(self, config, wav1, wav2, total_len, interpolate_rate=1) -> None:
        # Parameters that are required for calculation
        self.config = config 
        self.ref_signal = wav1
        self.ofdm_signal = wav2
        ## TODO: replace with frame_len
        self.total_len = total_len 
        self.start_point = np.inf
        self.interpolate_rate = interpolate_rate
    
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
    
    def correctFreq(self, rcv, freq_shift):
        """Correct the frequency offset in the received signal"""
        Ts = 1/self.config.USRP_CONF.SAMPLE_RATE
        # Might need to add - Ts
        t = np.arange(len(rcv)) * Ts 
        return rcv * np.exp(-1j*2*np.pi*freq_shift*t)

    def calcDoppler(self, rcv):
        """Calculate the Doppler spectrum of the received signal"""
        nyquist = self.config.USRP_CONF.SAMPLE_RATE / 2
        # Cutoff frequency (56 MHz / 2 = 28 MHz)
        cutoff = 28e6
        
        # Check if cutoff is possible
        if cutoff >= nyquist:
            # Cannot filter, use original signal
            f, Pxx = signal.welch(rcv, self.config.USRP_CONF.SAMPLE_RATE, nperseg=1024, return_onesided=False)
            return f, Pxx

        # Design the filter
        numtaps = 101
        taps = signal.firwin(numtaps, cutoff, fs=self.config.USRP_CONF.SAMPLE_RATE)

        # Apply the filter
        filtered_rcv = signal.lfilter(taps, 1.0, rcv)

        f, Pxx = signal.welch(filtered_rcv, self.config.USRP_CONF.SAMPLE_RATE, nperseg=1024, return_onesided=False)
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
            return np.inf # Or a large number, as it means pure LoS
            
        k_factor = los_power / nlos_power
        return 10 * np.log10(k_factor)

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
    
    # Shift the CIR by the offset value 
    # then crop the CIR to the size of the reference signal
    def adjustCIR(self, cir, ref, first_peak):
        cir = np.roll(cir, first_peak-ref)
        cir = cir[:self.total_len]
        return cir
    
    
    def process(self, r_time, rcv, vehicle_metric, tx_vehicle_metric=None, save_corr=True):
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
        
        # Coarse frequency estimation
        preamble_coarse = self.getPreamble(rcv, peaks)
        freq_shift_coarse = self.moose_alg(preamble_coarse, self.config.USRP_CONF.SAMPLE_RATE)
        rcv_coarse_corrected = self.correctFreq(rcv, freq_shift_coarse)

        # Fine frequency estimation
        xcorr_fine, _ = self.getCIR(rcv_coarse_corrected, self.ref_signal)
        peaks_fine = self.getPeaks(xcorr_fine)

        if len(peaks_fine) == 0:
            # Correction failed, use coarse estimate
            freq_shift = freq_shift_coarse
            rcv = rcv_coarse_corrected
            xcorr = xcorr_fine
            peaks = peaks_fine
        else:
            preamble_fine = self.getPreamble(rcv_coarse_corrected, peaks_fine)
            freq_shift_fine = self.moose_alg(preamble_fine, self.config.USRP_CONF.SAMPLE_RATE)
            
            freq_shift = freq_shift_coarse + freq_shift_fine
            
            # Correct original rcv with total frequency shift
            rcv = self.correctFreq(rcv, freq_shift)
            
            # Recalculate xcorr and peaks with the fully corrected rcv
            xcorr, _ = self.getCIR(rcv, self.ref_signal)
            peaks = self.getPeaks(xcorr)

        # For future reference

        if len(peaks) == 0:
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
        # --- End Interpolation Logic ---

        first_peak = peaks[0]
        metrics.start_point = first_peak
        metrics.detected = True
        
        metrics.freq_offset = freq_shift
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
        
        peaks = [p for p in peaks if p + _pr_len <= len(rcv)]

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
        metrics.snr = np.array([self.calcSNR(rcv[peak:peak+_pr_len], self.ref_signal) for peak in peaks])
        metrics.avgSnr = np.mean(metrics.snr)

        # 7 - Doppler
        f, Pxx = self.calcDoppler(rcv)
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
