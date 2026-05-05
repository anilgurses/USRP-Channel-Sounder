from dataclasses import dataclass, asdict, field
import numpy as np
from geopy.distance import geodesic
import time
import pandas as pd
from scipy import signal

from utils.constants import *
from utils.channel import *
from utils.vhcl_processor import *
from utils.antenna import *
from utils.usrp_calibration import (
    DEFAULT_TX_REF_DBM,
    populate_link_budget_config,
)

TO_THRESHOLD = 0.02
CIR_OFFSET = 1000 # Offset for the CIR delay
CIR_START = 300
CLIP_COMPONENT_THRESHOLD = 0.999
PATHLOSS_MIN_SNR_DB = -15.0
# A frame is only usable for absolute path loss if the ADC is effectively
# unsaturated. Clip-frac alone is too lenient (20% at the rail still clips
# every strong peak); clip-max catches single-sample overflow.
PATHLOSS_MAX_CLIP_FRAC = 0.02
# Clip-max is measured on the raw (pre-freq-correction) ADC samples; for
# signed 16-bit captures the max normalised component is 32767/32768 ≈
# 0.99997, so any value ≥ 1.0 indicates an impossible/overflowed sample
# (e.g. rounding overflow or wrap). Measured *after* frequency correction,
# max(|re|,|im|) is not rotation-invariant and can grow up to √2 times the
# raw magnitude 
PATHLOSS_MAX_CLIP_MAX = 1.0


def classify_saturation_status(clip_frac: float, clip_max: float) -> str:
    frac_over = bool(float(clip_frac) >= PATHLOSS_MAX_CLIP_FRAC)
    max_over = bool(float(clip_max) >= PATHLOSS_MAX_CLIP_MAX)
    if frac_over and max_over:
        return "both"
    if frac_over:
        return "clip_frac"
    if max_over:
        return "clip_max"
    return "clean"


# Link-budget constants
TX_REF_DBM = DEFAULT_TX_REF_DBM

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
    signal_power: np.ndarray
    avgSigPower: np.float32
    snr: np.ndarray
    avgSnr: np.float32
    freq_offset: np.float64
    path_loss: np.ndarray
    avg_pl: np.float16
    noise_power_dbfs: np.float32
    est_dist: np.float32
    peaks: np.ndarray
    orig_peaks: np.ndarray
    start_point: np.uint32
    clip_frac: np.float32
    clip_max: np.float32
    saturation_status: str
    pl_valid: bool
    
    aod_theta: np.float32
    aod_phi: np.float32
    aoa_theta: np.float32 
    aoa_phi: np.float32
    
    stage: str
    
    vehicle: VehicleMetric
    
    # TODO to be replaced with Channel class
    corr: np.ndarray
    corr_lags: np.ndarray = field(default_factory=lambda: np.array([]))
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
        return (
            f"SignalMetric: time={self.time}, center_freq={self.center_freq}, "
            f"dist={self.dist}, h_dist={self.h_dist}, v_dist={self.v_dist}, "
            f"wav_type={self.wav_type}, detected={self.detected}, snr={self.snr}, "
            f"rsrp={self.rsrp}, power={self.power}, avgPower={self.avgPower}, "
            f"freq_offset={self.freq_offset}, path_loss={self.path_loss}, "
            f"avg_pl={self.avg_pl}, shadowing={self.shadowing}, "
            f"multipath={self.multipath}, delay={self.delay}, "
            f"doppler_shift={self.doppler_shift}, est_dist={self.est_dist}, "
            f"peaks={self.peaks}, vehicle={self.vehicle}, corr={self.corr}, "
            f"save_corr={self.save_corr}"
        )

    __repr__ = __str__

    def __to_dict__(self):
        dct = asdict(self)
        v_dct = self.vehicle.__to_dict__()
        del v_dct["time"]
        del dct["vehicle"]
        dct.update(v_dct)
        return dct

    def to_scalar_dict(self):
        """Return only scalar values: lightweight dict for DataFrame construction.

        Excludes large numpy arrays (corr, power, snr, path_loss, peaks, etc.)
        that are not persisted to CSV, drastically reducing memory when results
        are collected from multiprocessing workers.
        """
        d = {
            'time': self.time, 'center_freq': self.center_freq,
            'dist': self.dist, 'h_dist': self.h_dist, 'v_dist': self.v_dist,
            'wav_type': self.wav_type, 'detected': self.detected,
            'avgPower': self.avgPower, 'avgSigPower': self.avgSigPower,
            'avgSnr': self.avgSnr,
            'freq_offset': self.freq_offset, 'avg_pl': self.avg_pl,
            'noise_power_dbfs': self.noise_power_dbfs,
            'est_dist': self.est_dist, 'start_point': self.start_point,
            'aod_theta': self.aod_theta, 'aod_phi': self.aod_phi,
            'aoa_theta': self.aoa_theta, 'aoa_phi': self.aoa_phi,
            'stage': self.stage, 'clip_frac': self.clip_frac,
            'clip_max': self.clip_max,
            'saturation_status': self.saturation_status,
            'pl_valid': self.pl_valid,
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
    def __init__(self, config, wav1, wav2, total_len) -> None:
        # Parameters that are required for calculation
        self.config = config
        self.ref_signal = wav1
        self.ofdm_signal = wav2
        ## TODO: replace with frame_len
        self.total_len = total_len
        self.start_point = np.inf
        self.seq_len = len(self.ref_signal)
        self.zc_repeat_count = max(
            int(round(float(self.total_len) / max(self.seq_len, 1))),
            1,
        )
    
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

    ## Digital power (dBFS): var() returns mean power E[|x|²], so use 10·log10.
    ## Optionally debias by the per-sample noise power σ²_w; at that point
    ## var(r) ≈ |α|²·mean(|s|²) + σ²_w, so subtracting σ²_w recovers the
    ## signal-only power.
    def calcPowerdBm(self, sig_of_int, noise_power_linear=0.0):
        power = float(np.var(sig_of_int))
        if noise_power_linear > 0.0:
            power -= float(noise_power_linear)
        return 10 * np.log10(max(power, 1e-13))

    def calcPathLossFromPowerDbfs(self, power_dbfs, tx_ref_dbm, rx_ref_dbm):
        return tx_ref_dbm + self.calcPowerdBm(self.ref_signal) - rx_ref_dbm - float(power_dbfs)

    def calcSignalPowerDbfsFromCorrPeak(self, corr_peak, noise_power_linear=0.0):
        seq_energy = np.sum(np.abs(self.ref_signal) ** 2) + 1e-30
        seq_len = max(len(self.ref_signal), 1)
        sig_power = (np.abs(corr_peak) ** 2) / (seq_energy * seq_len + 1e-30)
        # Matched-filter output bias: E[|corr_peak|²] contains a σ²_w·seq_energy
        # term, which becomes σ²_w/seq_len once normalized by (seq_energy·seq_len).
        # The processing gain (1/seq_len) makes this much smaller than the
        # variance-domain bias, but subtracting it is still the right thing.
        if noise_power_linear > 0.0:
            sig_power -= float(noise_power_linear) / seq_len
        return 10 * np.log10(max(sig_power, 1e-30))

    def estimateNoisePowerLinearFromCorr(self, xcorr, peak_samples):
        """Estimate per-sample noise power sigma^2_w from off-peak correlator bins.

        At a noise-only lag k, xcorr[k] = sum(w[n]·s*[n-k])_n is a sum of seq_len
        i.i.d. zero-mean complex terms, so E[|xcorr[k]|^2] = sigma^2_w · seq_energy.
        We mask out one sequence length around every detected peak, then use
        the median of |xcorr|^2 (robust against undetected peaks or multipath
        replicas) and convert median -> mean using the exponential-distribution
        factor ln(2).
        """
        if xcorr is None or len(xcorr) == 0:
            return 0.0

        seq_len = max(len(self.ref_signal), 1)
        seq_energy = float(np.sum(np.abs(self.ref_signal) ** 2))
        if seq_energy <= 0.0:
            return 0.0

        mask = np.ones(len(xcorr), dtype=bool)
        guard = seq_len
        for p in peak_samples:
            center = int(p) + seq_len - 1
            lo = max(0, center - guard)
            hi = min(len(xcorr), center + guard + 1)
            mask[lo:hi] = False

        off_peak_mag2 = np.abs(xcorr[mask]) ** 2
        if off_peak_mag2.size < 32:
            return 0.0

        noise_corr_energy = float(np.median(off_peak_mag2)) / np.log(2.0)
        return max(noise_corr_energy / seq_energy, 0.0)

    def calcClipStats(self, sig_of_int, threshold=CLIP_COMPONENT_THRESHOLD):
        comp_mag = np.maximum(np.abs(sig_of_int.real), np.abs(sig_of_int.imag))
        return float(np.mean(comp_mag >= threshold)), float(np.max(comp_mag))
    
    def calcResidualFitSNR(self, sig_of_interest, seq):
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

    def calcMatchedFilterSNR(self, signal_power_dbfs, noise_power_linear):
        if not np.isfinite(signal_power_dbfs) or noise_power_linear <= 0.0:
            return np.nan
        signal_power_linear = float(10.0 ** (float(signal_power_dbfs) / 10.0))
        if signal_power_linear <= 0.0:
            return np.nan
        return float(10.0 * np.log10(signal_power_linear / float(noise_power_linear)))
    
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
        # i'm jsut filterin out by setting out to 0
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
            return 40.0  # Cap at 40 dB. pure LoS (no measurable NLOS power)
            
        k_factor = los_power / nlos_power
        return 10 * np.log10(k_factor)

    def getPreamble(self, rcv, peak_samples):
        if len(peak_samples) == 0:
            return np.array([], dtype=np.complex64)
        preamble_len = max(2 * self.seq_len, 0)
        start = int(peak_samples[0])
        end = min(len(rcv), start + preamble_len)
        preamble = rcv[start:end]
        return preamble

    def _extract_zc_blocks(self, rcv, start_sample):
        if self.seq_len <= 0:
            return []

        blocks = []
        for rep_idx in range(self.zc_repeat_count):
            start = int(start_sample + rep_idx * self.seq_len)
            end = start + self.seq_len
            if start < 0 or end > len(rcv):
                break
            blocks.append(rcv[start:end])
        return blocks

    def estimateZCFreqOffset(self, rcv, peak_samples):
        if len(peak_samples) == 0:
            return np.nan

        blocks = self._extract_zc_blocks(rcv, int(peak_samples[0]))
        if len(blocks) < 2:
            return np.nan

        sample_rate = float(self.config.USRP_CONF.SAMPLE_RATE)
        estimates = []
        weights = []

        for lag in range(1, len(blocks)):
            pair_products = []
            for idx in range(len(blocks) - lag):
                prod = np.vdot(blocks[idx], blocks[idx + lag])
                if np.isfinite(prod.real) and np.isfinite(prod.imag):
                    pair_products.append(prod)
            if not pair_products:
                continue

            total_prod = np.sum(pair_products, dtype=np.complex128)
            weight = float(np.abs(total_prod))
            if weight <= 0.0:
                continue

            sample_delta = lag * self.seq_len
            estimates.append(
                float(np.angle(total_prod) * sample_rate / (2 * np.pi * sample_delta))
            )
            weights.append(weight)

        if not estimates:
            return np.nan
        return float(np.average(estimates, weights=weights))

    def estimateFreqOffset(self, rcv, peak_samples):
        if self.config.WAVEFORM == "ZC":
            zc_est = self.estimateZCFreqOffset(rcv, peak_samples)
            if np.isfinite(zc_est):
                return float(zc_est)

        preamble = self.getPreamble(rcv, peak_samples)
        if len(preamble) < 2:
            return 0.0
        return float(self.moose_alg(preamble, self.config.USRP_CONF.SAMPLE_RATE))
    
    def getCIR(self, rcv, ref, normalize=False):
        xcorr = signal.correlate(rcv, ref, mode="full", method="fft")
        lags = signal.correlation_lags(len(rcv), len(ref), mode="full")
        if normalize:
            peak = np.max(np.abs(xcorr))
            if peak > 0:
                xcorr = xcorr / peak

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
        sgnlMetric.signal_power = np.array([])
        sgnlMetric.avgSigPower = np.nan
        sgnlMetric.snr = np.array([])
        sgnlMetric.avgSnr = 0.0
        sgnlMetric.freq_offset = 0.0
        sgnlMetric.path_loss = np.array([])
        sgnlMetric.avg_pl = 0.0
        sgnlMetric.noise_power_dbfs = np.nan
        sgnlMetric.est_dist = 0.0
        sgnlMetric.peaks = np.array([])
        sgnlMetric.start_point = 0
        sgnlMetric.clip_frac = 0.0
        sgnlMetric.clip_max = 0.0
        sgnlMetric.saturation_status = "no_signal"
        sgnlMetric.pl_valid = False
        sgnlMetric.aod_theta = 0.0
        sgnlMetric.aod_phi = 0.0
        sgnlMetric.aoa_theta = 0.0
        sgnlMetric.aoa_phi = 0.0
        sgnlMetric.stage = ""
        sgnlMetric.vehicle = vehicle
        sgnlMetric.corr = np.array([])
        sgnlMetric.corr_lags = np.array([])
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
        metrics = SignalMetric()

        metrics.time = np.float32(r_time)
        metrics.center_freq = self.config.USRP_CONF.CENTER_FREQ / 1e6 # MHz
        metrics.wav_type = self.config.WAVEFORM

        # Keep a reference to the raw (pre-freq-correction) samples so clip
        # statistics are measured on the original ADC output. max(|re|,|im|)
        # is not rotation-invariant, so measuring it after correctFreq() can
        # inflate by up to root 2 and fail the saturation gate.
        rcv_raw = rcv

        # 1 - Detect the signal
        xcorr, xcorr_lags = self.getCIR(rcv, self.ref_signal)
        peaks = self.getPeaks(xcorr)
        peak_samples = self.corrPeaksToSampleIndices(peaks, len(rcv))
        peak_samples = peak_samples[
            peak_samples + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv)
        ]
        if len(peak_samples) == 0:
            ## Making sure that signal exists
            return self.zeroMetric(vehicle_metric)
        
        # Coarse frequency estimation
        freq_shift_coarse = self.estimateFreqOffset(rcv, peak_samples)
        rcv_coarse_corrected = self.correctFreq(rcv, freq_shift_coarse)

        # Fine frequency estimation
        xcorr_fine, xcorr_lags_fine = self.getCIR(rcv_coarse_corrected, self.ref_signal)
        peaks_fine = self.getPeaks(xcorr_fine)
        peak_samples_fine = self.corrPeaksToSampleIndices(peaks_fine, len(rcv_coarse_corrected))
        peak_samples_fine = peak_samples_fine[
            peak_samples_fine + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv_coarse_corrected)
        ]

        if len(peak_samples_fine) == 0:
            freq_shift = freq_shift_coarse
            rcv = rcv_coarse_corrected
            xcorr = xcorr_fine
            xcorr_lags = xcorr_lags_fine
            peaks = self.getPeaks(xcorr_fine, prm=20)  # lower prominence threshold
            peak_samples = self.corrPeaksToSampleIndices(peaks, len(rcv))
            peak_samples = peak_samples[
                peak_samples + self.config.WAV_OPTS.SEQ_LEN * 2 <= len(rcv)
            ]
        else:
            freq_shift_fine = self.estimateFreqOffset(
                rcv_coarse_corrected,
                peak_samples_fine,
            )
            
            freq_shift = freq_shift_coarse + freq_shift_fine
            
            # Correct original rcv with total frequency shift
            rcv = self.correctFreq(rcv, freq_shift)
            
            # Recalculate xcorr and peaks with the fully corrected rcv
            xcorr, xcorr_lags = self.getCIR(rcv, self.ref_signal)
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

        first_peak = int(peak_samples[0])
        metrics.start_point = first_peak
        metrics.detected = True
        
        metrics.freq_offset = freq_shift
        metrics.orig_peaks = peak_samples.copy()
        metrics.peaks = peak_samples.copy()
        
        if save_corr:
            metrics.corr = np.array(xcorr, copy=True)
            metrics.corr_lags = np.array(xcorr_lags, copy=True)
            metrics.save_corr = True
        else:
            metrics.corr = np.array([])
            metrics.corr_lags = np.array([])
            metrics.save_corr = False
        
        # 2 - Calculate the power of the signal
        _pr_len = len(self.ref_signal)

        peak_samples = [int(p) for p in peak_samples if p + _pr_len <= len(rcv)]

        # Estimate per-sample noise power sigma^2_w from off-peak matched-filter
        # bins, then debias both power estimators. The variance-based estimator
        # is biased by sigma^2_w directly; the correlation-based estimator only by
        # sigma^2_w/seq_len due to the matched-filter processing gain.
        noise_power_linear = self.estimateNoisePowerLinearFromCorr(xcorr, peak_samples)
        metrics.noise_power_dbfs = (
            float(10.0 * np.log10(noise_power_linear))
            if noise_power_linear > 0.0 else np.nan
        )

        metrics.power = np.array([
            self.calcPowerdBm(rcv[peak:peak+_pr_len], noise_power_linear)
            for peak in peak_samples
        ])
        corr_peak_idxs = np.clip(
            np.asarray(peak_samples, dtype=np.int64) + (_pr_len - 1),
            0, len(xcorr) - 1,
        )
        metrics.signal_power = np.array([
            self.calcSignalPowerDbfsFromCorrPeak(xcorr[idx], noise_power_linear)
            for idx in corr_peak_idxs
        ])
        clip_stats = [self.calcClipStats(rcv_raw[peak:peak+_pr_len]) for peak in peak_samples]
        clip_fracs = np.array([stat[0] for stat in clip_stats], dtype=float)
        clip_maxes = np.array([stat[1] for stat in clip_stats], dtype=float)
            
        if len(metrics.power) > 0:
            metrics.avgPower = np.mean(metrics.power)
        if len(metrics.signal_power) > 0:
            metrics.avgSigPower = np.mean(metrics.signal_power)
        if len(clip_fracs) > 0:
            metrics.clip_frac = float(np.mean(clip_fracs))
        if len(clip_maxes) > 0:
            metrics.clip_max = float(np.max(clip_maxes))
        
        if getattr(metrics, 'avgPower', None) is None:
            metrics.avgPower = np.nan
        if getattr(metrics, 'avgSigPower', None) is None:
            metrics.avgSigPower = np.nan
        
        # 3 - Calculate path loss using the campaign-specific link-budget
        # for A2G: lw1 TX, pn6 RX for A2A: pn3: RX pn6: TX
        populate_link_budget_config(self.config, default_tx_ref_dbm=TX_REF_DBM)

        metrics.path_loss = np.array([
            self.calcPathLossFromPowerDbfs(power_dbfs, self.config.TX_REF_DBM, self.config.RX_REF_DBM)
            for power_dbfs in metrics.signal_power
        ])
        if len(metrics.path_loss) > 0:
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
        if noise_power_linear > 0.0:
            metrics.snr = np.array([
                self.calcMatchedFilterSNR(power_dbfs, noise_power_linear)
                for power_dbfs in metrics.signal_power
            ], dtype=float)
        else:
            metrics.snr = np.array([
                self.calcResidualFitSNR(rcv[peak:peak+_pr_len], self.ref_signal)
                for peak in peak_samples
            ], dtype=float)
        metrics.avgSnr = np.mean(metrics.snr)

        metrics.saturation_status = classify_saturation_status(
            metrics.clip_frac, metrics.clip_max
        )

        metrics.pl_valid = bool(
            np.isfinite(metrics.avg_pl)
            and np.isfinite(metrics.avgSnr)
            and (metrics.avgSnr > PATHLOSS_MIN_SNR_DB)
            and (metrics.saturation_status == "clean")
        )

        # 7 - Wideband PSD of this frame 
        f, Pxx = self.calcWidebandPSD(rcv)
        metrics.doppler_shift = metrics.freq_offset
        metrics.doppler_spectrum = Pxx

        # 8 - RMS Delay Spread and K-Factor
        # Crop the CIR around the main peak (~2 us window) for delay-spread stats.
        max_peak_index = np.argmax(np.abs(xcorr))
        window_size = 100
        half_window = window_size // 2

        start_crop = max(0, max_peak_index - half_window)
        end_crop = min(len(xcorr), max_peak_index + half_window)
        cropped_cir = xcorr[start_crop:end_crop]

        # Estimate noise floor from AFTER the signal window (avoids start-edge effects).
        noise_start = end_crop + 100
        noise_end = noise_start + 200
        if noise_end < len(xcorr):
            noise_floor_pdp = np.mean(np.abs(xcorr[noise_start:noise_end])**2)
        else:
            # Fallback: use region before the peak if not enough samples after
            noise_end_alt = max(0, start_crop - 100)
            noise_start_alt = max(0, noise_end_alt - 200)
            if noise_start_alt < noise_end_alt:
                noise_floor_pdp = np.mean(np.abs(xcorr[noise_start_alt:noise_end_alt])**2)
            else:
                noise_floor_pdp = np.percentile(np.abs(cropped_cir)**2, 10)

        metrics.rms_delay_spread = self.calcRMSDelaySpread(
            cropped_cir, self.config.USRP_CONF.SAMPLE_RATE, noise_floor=noise_floor_pdp,
        )
        metrics.k_factor = self.calcKFactor(cropped_cir, noise_floor=noise_floor_pdp)

        return metrics
