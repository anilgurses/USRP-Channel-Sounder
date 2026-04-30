cimport cython
import uhd

from scipy.signal import correlate
import time
from time import sleep
from scipy.io import savemat
from queue import Empty
from BufferToProcess import BufferToProcess
from WaveformGenerator import Waveform
from Scheduler import PpsSlotScheduler
from datetime import datetime
import os
import math
# from dronekit import connect
import yaml
import serial

import numpy as np
cimport numpy as np

np.import_array()


# Saturation thresholds for fc32 samples 
SAT_MAG_THR = 0.95           # |x| above this counts as near-saturation
SAT_PCT_WARN = 1.0           # % of samples > SAT_MAG_THR triggers SATURATED flag
SAT_MIN_COUNT = 8            # Ignore isolated spikes / stale samples.

# Packet detection: peak / median(noise) ratio in dB to declare DETECTED
DETECT_SNR_DB = 10.0
DETECT_NORM_PEAK = 0.45

# PPS-drift tolerance: deviation (in seconds) from an integer-second interval
# beyond which we warn. With external PPS this should stay well under 1us.
PPS_DRIFT_WARN = 1.0e-4

# ctypedef Config ConfigType

class Receiver:
    # cdef ConfigType config
    # cdef int plot

    def __init__(self, config, plot):
        self.config = config
        self.plot = plot
        self.output_type = self.config.RX.OUTPUT_TYPE

    def _queue_depth(self, que):
        try:
            return que.qsize()
        except Exception:
            return -1

    def _format_optional_us(self, value):
        if value is None:
            return "n/a"
        return "%.3f" % (value * 1e6)

    def _dashboard_put(self, que, payload):
        if que is None:
            return
        try:
            que.put(payload)
        except Exception:
            pass

    def _log_rx_burst(self, logger, idx, status, scheduled_rx_start,
                      actual_rx_start, actual_last_sample, host_issue_time,
                      host_recv_done, last_pps_usrp_time,
                      rx_start_minus_last_pps, pps_drift, pps_stale,
                      num_rx_samps, expected_samps, queue_depth):
        schedule_error_us = (actual_rx_start - scheduled_rx_start) * 1e6
        rx_duration_ms = (
            (actual_last_sample - actual_rx_start) * 1e3
            if actual_last_sample >= actual_rx_start else 0.0
        )
        issue_lead_ms = (scheduled_rx_start - host_issue_time) * 1e3
        recv_wait_ms = (host_recv_done - host_issue_time) * 1e3
        missing_samps = max(0, expected_samps - num_rx_samps)
        pps_drift_us = self._format_optional_us(pps_drift)

        logger.info(
            (
                "RX_BURST idx=%d status=%s samps=%d/%d missing=%d "
                "scheduled=%.9f start=%.9f last=%.9f "
                "schedule_error=%.3fus rx_duration=%.3fms "
                "issue_lead=%.3fms recv_wait=%.3fms "
                "last_pps=%.9f rx_pps_offset=%.9f pps_drift=%sus "
                "pps_stale=%s queue_depth=%d"
            ),
            idx,
            status,
            num_rx_samps,
            expected_samps,
            missing_samps,
            scheduled_rx_start,
            actual_rx_start,
            actual_last_sample,
            schedule_error_us,
            rx_duration_ms,
            issue_lead_ms,
            recv_wait_ms,
            last_pps_usrp_time,
            rx_start_minus_last_pps,
            pps_drift_us,
            "yes" if pps_stale else "no",
            queue_depth,
        )


    def coarse_freq_est(self, rec_sig, num_lag, raised_buff):
        scalingFactor = self.config.SAMPLE_RATE / ((num_lag + 1) * 2 * np.pi)

        # Raise the signal to Mth power
        raisedSignal = rec_sig ** 2
        raisedSignalBuffer = raised_buff

        autocorrsum = np.complex64(0)

        temp = np.concatenate((raisedSignalBuffer[0],raisedSignal[0]))
        autocorrsum = np.sum(np.correlate(temp,raisedSignal[0]))

        freqOff = scalingFactor * np.angle(autocorrsum)
        raisedSignalBuffer = raisedSignal[:,rec_sig.size-num_lag:]

        return freqOff, raisedSignalBuffer

    # Not needed
    def preamble_detector(self, sig, seq, filter_states, threshold = None):
        cdef np.ndarray xcorr = correlate(sig[0], seq, mode="full")
        #xcorr, new_filter_states = lfilter(seq, 1, sig[0], zi=filter_states)
        xcorr = np.absolute(xcorr)

        if not threshold:
            # It's just a dummy value
            idPrEnd = np.where(xcorr >= 1)
        else:
            idPrEnd = np.where(xcorr >= threshold-self.config.THR_OFF)

        return idPrEnd[0], filter_states

    def calc_power(self, sig, logger):
        stats = self.signal_health(sig, logger)
        return stats["power_dbm"]

    def signal_health(self, sig, logger, emit_log=True, rx_time=None):
        # sig is shape (1, N) complex64; restrict to one burst window when long.
        x = sig[0, :self.calc_window] if sig.shape[1] >= self.calc_window else sig[0]
        if x.size == 0:
            if emit_log:
                logger.warn("RX_HEALTH idx=%d empty buffer", self.frame_idx)
            return {
                "power_dbfs": float("-inf"),
                "peak_dbfs": float("-inf"),
                "power_dbm": float("-inf"),
                "max_i": 0.0,
                "max_q": 0.0,
                "sat_pct": 0.0,
                "crest_db": 0.0,
                "saturated": False,
            }

        abs_x = np.abs(x)
        mean_power = float(np.mean(abs_x ** 2))
        peak = float(np.max(abs_x))
        rms = float(np.sqrt(np.mean(abs_x ** 2)))
        max_i = float(np.max(np.abs(x.real)))
        max_q = float(np.max(np.abs(x.imag)))

        sat_count = int(np.count_nonzero(abs_x > SAT_MAG_THR))
        sat_pct = 100.0 * sat_count / x.size
        saturated = sat_count >= SAT_MIN_COUNT and sat_pct > SAT_PCT_WARN

        power_dbfs = 10.0 * math.log10(mean_power) if mean_power > 0 else float("-inf")
        peak_dbfs = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
        crest_db = 20.0 * math.log10(peak / rms) if rms > 0 and peak > 0 else 0.0
        power_dbm = power_dbfs + self.config.CAL.RX_REF

        flag = "SATURATED" if saturated else "OK"
        rx_time_text = "n/a" if rx_time is None else "%.9f" % float(rx_time)
        if emit_log:
            logger.info(
                (
                    "RX_HEALTH idx=%d t_rx=%s %s mean=%.2fdBFS peak=%.2fdBFS pwr=%.2fdBm "
                    "max_i=%.3f max_q=%.3f sat_pct=%.4f%% crest=%.2fdB"
                ),
                self.frame_idx, rx_time_text, flag, power_dbfs, peak_dbfs, power_dbm,
                max_i, max_q, sat_pct, crest_db,
            )

        return {
            "power_dbfs": power_dbfs,
            "peak_dbfs": peak_dbfs,
            "power_dbm": power_dbm,
            "max_i": max_i,
            "max_q": max_q,
            "sat_pct": sat_pct,
            "crest_db": crest_db,
            "saturated": saturated,
        }

    def detect_packet(self, sig, ref, logger, rx_time=None):
        x = sig[0]
        if x.size < ref.size or ref.size == 0:
            logger.warn("RX_DETECT idx=%d skip: buf=%d ref=%d",
                        self.frame_idx, x.size, ref.size)
            return {
                "detected": False, "peak": 0.0, "peak_idx": -1,
                "first_peak_idx": -1, "snr_db": float("-inf"),
                "norm_peak": 0.0,
            }

        xcorr = np.abs(correlate(x, ref, mode="valid", method="fft"))
        peak_idx = int(np.argmax(xcorr))
        peak_val = float(xcorr[peak_idx])

        ref_energy = float(np.sum(np.abs(ref) ** 2))
        win_energy = np.convolve(
            np.abs(x) ** 2,
            np.ones(ref.size, dtype=np.float32),
            mode="valid",
        )
        denom = math.sqrt(ref_energy * float(win_energy[peak_idx]))
        norm_peak = peak_val / denom if denom > 0 else 0.0

        # When there are repeated ZCs in the burst, multiple peaks appear at
        # equal spacing. Take the leftmost one above 70% of the global max
        # so that downstream OFDM offset arithmetic anchors on ZC #1
        first_peak_idx = peak_idx
        if peak_val > 0:
            above = np.where(xcorr >= 0.7 * peak_val)[0]
            if above.size > 0:
                first_peak_idx = int(above[0])

        # Noise floor: median outside a guard region around any strong peak.
        guard = ref.size
        mask = np.ones(xcorr.size, dtype=bool)
        for p in (peak_idx, first_peak_idx):
            lo = max(0, p - guard)
            hi = min(xcorr.size, p + guard)
            mask[lo:hi] = False
        if np.any(mask):
            noise_med = float(np.median(xcorr[mask]))
        else:
            noise_med = 0.0

        snr_db = 20.0 * math.log10(peak_val / noise_med) if (noise_med > 0 and peak_val > 0) else float("inf")
        detected = bool(
            snr_db >= DETECT_SNR_DB
            and norm_peak >= DETECT_NORM_PEAK
            and peak_val > 0
        )

        rx_time_text = "n/a" if rx_time is None else "%.9f" % float(rx_time)
        logger.info(
            (
                "RX_DETECT idx=%d t_rx=%s %s peak=%.4f peak_idx=%d first_peak_idx=%d "
                "noise_med=%.5f corr_snr=%.1fdB norm_peak=%.3f"
            ),
            self.frame_idx, rx_time_text, "DETECTED" if detected else "MISSED",
            peak_val, peak_idx, first_peak_idx, noise_med, snr_db, norm_peak,
        )

        return {
            "detected": detected,
            "peak": peak_val,
            "peak_idx": peak_idx,
            "first_peak_idx": first_peak_idx,
            "snr_db": snr_db,
            "norm_peak": norm_peak,
        }

    def estimate_ofdm_channel(self, sig, first_peak_idx, logger):
        """
        Pilot-based per-subcarrier channel estimate, RSRP and SNR.

        Layout from WaveformGenerator: first_peak_idx points at the start of
        ZC #1. The OFDM CP begins n_rep*(seq_len+guard) samples later (one
        guard between each ZC pair plus one guard before OFDM = n_rep guards
        for n_rep ZCs). The OFDM symbol body starts cp_len samples after that.
        """
        n_fft = self._ofdm_n_fft
        cp_len = self._ofdm_cp_len
        positions = self._ofdm_pilot_pos
        x_known = self._ofdm_pilot_val

        seq_len = int(self.config.WAV_OPTS.SEQ_LEN)
        guard = int(getattr(self.config.WAV_OPTS, "GUARD_LEN_SAMPS", 100))
        n_rep = int(getattr(self.config.WAV_OPTS, "ZC_NUM_REPEATS", 4))

        offset_to_ofdm = n_rep * (seq_len + guard) + cp_len
        sym_start = first_peak_idx + offset_to_ofdm
        sym_end = sym_start + n_fft

        sig1d = sig[0]
        if sym_end > sig1d.size or sym_start < 0:
            logger.warn(
                "RX_OFDM idx=%d skip: peak=%d sym=[%d,%d) buf=%d",
                self.frame_idx, first_peak_idx, sym_start, sym_end, sig1d.size,
            )
            return None

        rx_sym = sig1d[sym_start:sym_end]
        Y = np.fft.fft(rx_sym, n=n_fft)
        H_hat = Y[positions] / x_known                     # per-pilot channel
        sig_lin = float(np.mean(np.abs(H_hat) ** 2))

        # Smoothed-residual SNR: low-pass H_hat in frequency, take residual
        # as noise. Window of 5 bins is a good compromise for ~500-bin grid.
        win = 5
        kernel = np.ones(win, dtype=np.complex64) / win
        H_smooth = np.convolve(H_hat, kernel, mode="same")
        # Trim window halves where convolution edges bias the estimate.
        edge = win // 2
        if H_hat.size > 2 * edge:
            resid = H_hat[edge:-edge] - H_smooth[edge:-edge]
            sig_smooth_lin = float(np.mean(np.abs(H_smooth[edge:-edge]) ** 2))
        else:
            resid = H_hat - H_smooth
            sig_smooth_lin = float(np.mean(np.abs(H_smooth) ** 2))
        noise_lin = float(np.mean(np.abs(resid) ** 2))

        rsrp_dbfs = 10.0 * math.log10(sig_lin) if sig_lin > 0 else float("-inf")
        rsrp_dbm = rsrp_dbfs + self.config.CAL.RX_REF
        snr_db = (10.0 * math.log10(sig_smooth_lin / noise_lin)
                  if noise_lin > 0 and sig_smooth_lin > 0 else float("inf"))

        logger.info(
            (
                "RX_OFDM idx=%d sym=[%d,%d) bins=%d RSRP=%.2fdBFS / %.2fdBm "
                "SNR=%.2fdB"
            ),
            self.frame_idx, sym_start, sym_end, positions.size,
            rsrp_dbfs, rsrp_dbm, snr_db,
        )

        return {
            "rsrp_dbfs": rsrp_dbfs,
            "rsrp_dbm": rsrp_dbm,
            "snr_db": snr_db,
            "n_active_bins": int(positions.size),
            "ofdm_sym_start": int(sym_start),
        }

    def process_recv_data(self, que, pwr_queue, logger, terminate, dashboard_enabled=False):
        # cdef int modOrder = 2
        # cdef float maxToneOffset = self.config.MAX_FREQ_OFF * modOrder
        # cdef int numLag = int(np.round(self.config.USRP_CONF.SAMPLE_RATE/maxToneOffset)) - 1
        # cuFreqOff = 0

        wv = Waveform(self.config)
        waveform = wv.create_waveform()
        self.calc_window = waveform.shape[0] * 2
        self.frame_idx = 0

        # Build a short reference for packet detection. Prefer one ZC period
        # so the correlator stays cheap and the peak corresponds to the start
        # of the burst.
        if self.config.WAVEFORM == "ZC":
            self.detect_ref = wv.create_zadoff_chu()
        else:
            ref_len = min(int(waveform.shape[0]), 4096)
            self.detect_ref = waveform[:ref_len].astype(np.complex64)

        # Pilot layout for OFDM channel estimator (matches transmit side).
        try:
            (self._ofdm_pilot_pos, self._ofdm_pilot_val,
             self._ofdm_n_fft, self._ofdm_cp_len) = wv.get_ofdm_pilots()
        except Exception as ex:
            logger.warn("OFDM pilot layout unavailable: %s", ex)
            self._ofdm_pilot_pos = np.empty(0, dtype=np.int32)
            self._ofdm_pilot_val = np.empty(0, dtype=np.complex64)
            self._ofdm_n_fft = 0
            self._ofdm_cp_len = 0

        # bfrRx = BufferToProcess(int(self.config.USRP_CONF.SAMPLE_RATE * 1 / self.config.RX.DURATION))

        # Initial filter taps
        # cdef np.ndarray prb_filter_states = np.zeros(prb_intr.size-1, dtype=np.complex64)

        out_dir = f"../measurements/{datetime.today().strftime('%Y-%m-%d_%H_%M')}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(f"{out_dir}/config.yaml", "w") as f:  # Save config next to measurement.
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        while not terminate.is_set():
            try:
                recv_buffer, num_rx_samps, rcv_time, gps_info, time_info, pps_info = que.get(timeout=0.5)
            except Empty:
                continue

            buff = recv_buffer
            self.frame_idx += 1

            health = None
            detect = None
            ofdm_meas = None

            if self.config.RX.POWER_CALC or self.config.RX.PL_CALC or dashboard_enabled:
                health = self.signal_health(
                    buff,
                    logger,
                    emit_log=self.config.RX.POWER_CALC,
                    rx_time=rcv_time,
                )

            detect = self.detect_packet(buff, self.detect_ref, logger, rx_time=rcv_time)

            # Pilot-based RSRP/SNR when packet detected and the
            # OFDM symbol fits in the buffer. RSRP is the preferred input to
            # the path-loss calculation: it excludes guard / silence and is
            # measured on known reference symbols.
            if (detect.get("detected") and self._ofdm_n_fft > 0
                    and self._ofdm_pilot_pos.size > 0):
                ofdm_meas = self.estimate_ofdm_channel(
                    buff, detect["first_peak_idx"], logger,
                )

            if self.config.RX.PL_CALC:
                # PL_dB = TX_EIRP_dBm - RX_dBm 
                rx_dbm = None
                src = None
                pl_db = None
                if ofdm_meas is not None and ofdm_meas["rsrp_dbm"] != float("-inf"):
                    rx_dbm = ofdm_meas["rsrp_dbm"]
                    src = "rsrp"
                elif health is not None and health["power_dbm"] != float("-inf"):
                    rx_dbm = health["power_dbm"]
                    src = "wideband"
                if rx_dbm is not None:
                    pl_db = self.config.CAL.TX_REF - rx_dbm
                    logger.info(
                        "RX_PL idx=%d src=%s tx=%.2fdBm rx=%.2fdBm pl=%.2fdB",
                        self.frame_idx, src, self.config.CAL.TX_REF, rx_dbm, pl_db,
                    )
            else:
                pl_db = None

            if dashboard_enabled:
                self._dashboard_put(
                    pwr_queue,
                    {
                        "kind": "rx_metrics",
                        "idx": self.frame_idx,
                        "rx_time": rcv_time,
                        "power_dbfs": health.get("power_dbfs") if health else None,
                        "peak_dbfs": health.get("peak_dbfs") if health else None,
                        "power_dbm": health.get("power_dbm") if health else None,
                        "crest_db": health.get("crest_db") if health else None,
                        "sat_pct": health.get("sat_pct") if health else None,
                        "sat_status": (
                            "SATURATED" if health and health.get("saturated") else "OK"
                        ),
                        "detect_status": (
                            "DETECTED"
                            if detect and detect.get("detected")
                            else (
                                "NO SIGNAL"
                                if detect and detect.get("norm_peak", 0.0) < DETECT_NORM_PEAK
                                else "MISSED"
                            )
                        ),
                        "corr_snr_db": detect.get("snr_db") if detect else None,
                        "norm_peak": detect.get("norm_peak") if detect else None,
                        "rsrp_dbm": (
                            ofdm_meas.get("rsrp_dbm") if ofdm_meas is not None else None
                        ),
                        "ofdm_snr_db": (
                            ofdm_meas.get("snr_db") if ofdm_meas is not None else None
                        ),
                        "pl_db": pl_db,
                    },
                )

            if self.config.RX.OUTPUT_TYPE == "npz":
                np.savez(f"{out_dir}/received_{rcv_time}.npz",
                         rcv    =   buff,
                         ref   =  waveform,
                         rx_time  = rcv_time,
                         gps_info   =  gps_info,
                         time_info  =  time_info,
                         pps_info   =  pps_info,
                         health     =  health if health is not None else {},
                         detect     =  detect if detect is not None else {},
                         ofdm_meas  =  ofdm_meas if ofdm_meas is not None else {},
                         allow_pickle = True
                        )
            elif self.config.RX.OUTPUT_TYPE == "mat":
                savemat(f"{out_dir}/received_{rcv_time}.mat",
                        {
                            "buff":buff,
                            "ref":waveform,
                            "rx_time": rcv_time,
                            "gps_info": gps_info,
                            "time_info": time_info,
                            "pps_info": pps_info,
                            "health": health if health is not None else {},
                            "detect": detect if detect is not None else {},
                            "ofdm_meas": ofdm_meas if ofdm_meas is not None else {},
                        })


    def receive(self, usrp, rx_stream, logger, rcv_queue, status_queue, terminate, args):
        gps = None
        gps_t = None

        if self.config.GPS.ENABLED:
            if self.config.GPS.SOURCE == "gnss":
                from ublox_gps import UbloxGps
                port = serial.Serial(self.config.GPS.DIR, baudrate=38400, timeout=1)
                gps_obj = UbloxGps(port)
                gps = gps_obj.geo_coords
                gps_t = gps_obj.date_time
            # elif self.config.GPS_SOURCE == "vehicle":
            #     gps_s = connect('/dev/ttyACM0')
            #     gps = gps_s.location.global_relative_frame


        max_samps_per_packet = int(self.config.USRP_CONF.SAMPLE_RATE * self.config.RX.DURATION)
        recv_buffer = np.empty((1, max_samps_per_packet), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        # Start continuous stream.
        # Craft and send the Stream Command
        rate = usrp.get_rx_rate()

        had_an_overflow = False
        last_overflow = uhd.types.TimeSpec(0)

        num_rx_dropped = 0
        burst_index = 0
        last_rx_event = None
        configured_log_every = getattr(self.config, "LOG_EVERY", None)
        if configured_log_every is None:
            log_every = max(1, int(getattr(self.config, "PERIOD", 1)) * 5)
        else:
            log_every = max(1, int(configured_log_every))

        # PPS-drift tracking
        prev_pps_time = None
        prev_pps_burst = None
        pps_stale_count = 0

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        time.sleep(1)
        logger.info(
            (
                "RX_STREAM_CONFIG rate=%.3fHz duration=%.6fs "
                "request_samps=%d period=%.3fHz log_every=%d "
                "clock_ref=%s pps_ref=%s"
            ),
            rate,
            float(self.config.RX.DURATION),
            max_samps_per_packet,
            float(self.config.PERIOD),
            log_every,
            self.config.USRP_CONF.CLK_REF,
            self.config.USRP_CONF.PPS_REF,
        )

        period = self.config.PERIOD
        scheduler = PpsSlotScheduler(
            period,
            usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY
        )
        inc_sec = 1 / period
        next_slot_index = 0
        logger.info(
            "RX_SCHED_SYNC epoch=%.9f period=%.3fHz slots_per_sec=%d slot=%.9fs",
            scheduler.epoch_s,
            float(period),
            scheduler.slots_per_second,
            scheduler.slot_s,
        )

        while not terminate.is_set():
            try:
                burst_index += 1
                scheduled_rx_start = scheduler.time_for_index(next_slot_index)
                usrp_time = usrp.get_time_now().get_real_secs()
                if scheduled_rx_start <= usrp_time:
                    previous_slot_index = next_slot_index
                    next_slot_index = scheduler.next_index_after(
                        usrp_time,
                        min_index=next_slot_index + 1,
                    )
                    skip_count = next_slot_index - previous_slot_index
                    scheduled_rx_start = scheduler.time_for_index(next_slot_index)
                    logger.warn(
                        "RX_SCHED_SKIP idx=%d skipped=%d usrp_time=%.9f new_time_rx=%.9f"
                        " slot=%d (TX/RX burst alignment lost)",
                        burst_index, skip_count, usrp_time, scheduled_rx_start,
                        next_slot_index,
                    )

                stream_cmd.stream_now = False
                stream_cmd.num_samps = max_samps_per_packet
                stream_cmd.time_spec = uhd.types.TimeSpec(scheduled_rx_start)
                host_issue_time = usrp.get_time_now().get_real_secs()
                rx_stream.issue_stream_cmd(stream_cmd)

                # The difference between scheduled time and usrp time can't be bigger than 1s
                num_rx_samps = rx_stream.recv(recv_buffer, metadata, 1)
                host_recv_done = usrp.get_time_now().get_real_secs()
                actual_rx_start = metadata.time_spec.get_real_secs()
                if num_rx_samps > 0:
                    actual_last_sample = actual_rx_start + (num_rx_samps - 1) / rate
                else:
                    actual_last_sample = actual_rx_start

                last_pps_usrp_time = usrp.get_time_last_pps().get_real_secs()
                pps_delta = None
                pps_drift = None
                pps_stale = False

                if prev_pps_time is not None:
                    if last_pps_usrp_time == prev_pps_time:
                        # PPS hasn't ticked since the previous frame. Expected
                        # when frame rate > 1 Hz, but if it persists across many
                        # bursts (>= 2x PERIOD) the external PPS may be missing.
                        pps_stale_count += 1
                        threshold = max(2, 2 * int(getattr(self.config, "PERIOD", 1)))
                        if pps_stale_count >= threshold:
                            pps_stale = True
                            logger.warn(
                                "RX_PPS_STALE idx=%d last_pps=%.9f stale_for=%d frames",
                                burst_index, last_pps_usrp_time, pps_stale_count,
                            )
                    else:
                        pps_delta = last_pps_usrp_time - prev_pps_time
                        expected = float(round(pps_delta))
                        pps_drift = pps_delta - expected
                        pps_stale_count = 0
                        if abs(pps_drift) > PPS_DRIFT_WARN:
                            logger.warn(
                                (
                                    "RX_PPS_DRIFT idx=%d last_pps=%.9f prev=%.9f "
                                    "delta=%.9fs drift=%.3fus"
                                ),
                                burst_index, last_pps_usrp_time, prev_pps_time,
                                pps_delta, pps_drift * 1e6,
                            )

                pps_info = {
                    "last_pps_usrp_time": last_pps_usrp_time,
                    "rx_start_minus_last_pps": actual_rx_start - last_pps_usrp_time,
                    "frame_rx_start_usrp_time": actual_rx_start,
                    "prev_last_pps_usrp_time": prev_pps_time,
                    "pps_interval_s": pps_delta,
                    "pps_drift_s": pps_drift,
                    "pps_stale": pps_stale,
                    "pps_stale_count": pps_stale_count,
                }

                prev_pps_time = last_pps_usrp_time
                prev_pps_burst = burst_index

            except RuntimeError as ex:
                logger.err("Runtime error in receive: %s", ex)
                return

            usrp_time = usrp.get_time_now().get_real_secs()
            next_slot_index += 1

            if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                schedule_error_us = (actual_rx_start - scheduled_rx_start) * 1e6
                rx_duration_ms = (
                    (actual_last_sample - actual_rx_start) * 1e3
                    if actual_last_sample >= actual_rx_start else 0.0
                )
                issue_lead_ms = (scheduled_rx_start - host_issue_time) * 1e3
                recv_wait_ms = (host_recv_done - host_issue_time) * 1e3
                missing_samps = max(0, max_samps_per_packet - num_rx_samps)
                pps_drift_us = (
                    pps_info["pps_drift_s"] * 1e6
                    if pps_info["pps_drift_s"] is not None else None
                )

                if burst_index == 1 or burst_index % log_every == 0:
                    self._log_rx_burst(
                        logger,
                        burst_index,
                        "ok" if num_rx_samps == max_samps_per_packet else "short",
                        scheduled_rx_start,
                        actual_rx_start,
                        actual_last_sample,
                        host_issue_time,
                        host_recv_done,
                        pps_info["last_pps_usrp_time"],
                        pps_info["rx_start_minus_last_pps"],
                        pps_info["pps_drift_s"],
                        pps_info["pps_stale"],
                        num_rx_samps,
                        max_samps_per_packet,
                        self._queue_depth(rcv_queue),
                    )
                self._dashboard_put(
                    status_queue,
                    {
                        "kind": "rx_burst",
                        "idx": burst_index,
                        "status": (
                            "ok" if num_rx_samps == max_samps_per_packet else "short"
                        ),
                        "sample_text": "%d/%d" % (num_rx_samps, max_samps_per_packet),
                        "num_rx_samps": num_rx_samps,
                        "expected_samps": max_samps_per_packet,
                        "missing_samps": missing_samps,
                        "scheduled": scheduled_rx_start,
                        "start": actual_rx_start,
                        "last": actual_last_sample,
                        "schedule_error_us": schedule_error_us,
                        "rx_duration_ms": rx_duration_ms,
                        "issue_lead_ms": issue_lead_ms,
                        "recv_wait_ms": recv_wait_ms,
                        "last_pps": pps_info["last_pps_usrp_time"],
                        "rx_pps_offset": pps_info["rx_start_minus_last_pps"],
                        "pps_drift_us": pps_drift_us,
                        "pps_stale": "yes" if pps_info["pps_stale"] else "no",
                        "queue_depth": self._queue_depth(rcv_queue),
                    },
                )
                last_rx_event = (
                    burst_index,
                    scheduled_rx_start,
                    actual_rx_start,
                    actual_last_sample,
                    host_issue_time,
                    host_recv_done,
                    num_rx_samps,
                    pps_info["last_pps_usrp_time"],
                    pps_info["rx_start_minus_last_pps"],
                    pps_info["pps_drift_s"],
                    pps_info["pps_stale"],
                )

                fr = None
                if gps:
                    fr = gps() if callable(gps) else gps

                gps_time = None
                if gps_t:
                    gps_time = gps_t() if callable(gps_t) else gps_t

                gps_info = {
                        "lat":  0.0,
                        "lon":  0.0,
                        "alt":  0.0,
                        "speed": 0.0,
                        "hdg": 0.0
                        }

                time_info = time.time()

                if gps_time is not None:
                    time_info = (
                        f"{gps_time.year}/{gps_time.month}/{gps_time.day} "
                        f"{gps_time.hour}:{gps_time.min}:{gps_time.sec}"
                    )

                if fr is not None:
                    gps_info["lat"] = fr.lat
                    gps_info["lon"] = fr.lon
                    gps_info["alt"] = fr.height
                    gps_info["hsea"] = fr.hMSL
                    gps_info["speed"] = fr.magAcc
                    gps_info["hdg"] = fr.headVeh

                # Add gps info here
                rcv_queue.put(
                    (
                        recv_buffer[:, :num_rx_samps].copy(),
                        num_rx_samps,
                        actual_rx_start,
                        gps_info,
                        time_info,
                        pps_info,
                    )
                )

                # Reset the overflow flag
                if had_an_overflow:
                    had_an_overflow = False
                    num_rx_dropped += (metadata.time_spec - last_overflow).to_ticks(rate)

            elif metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                had_an_overflow = True
                self._dashboard_put(
                    status_queue,
                    {
                        "kind": "event",
                        "level": "WARN",
                        "message": "idx=%d overflow scheduled=%.9f" % (
                            burst_index,
                            scheduled_rx_start,
                        ),
                    },
                )
                logger.warn(
                    (
                        "RX_BURST_ERR idx=%d status=overflow scheduled=%.9f "
                        "issue_time=%.9f recv_done=%.9f"
                    ),
                    burst_index,
                    scheduled_rx_start,
                    host_issue_time,
                    host_recv_done,
                )

                last_overflow = uhd.types.TimeSpec(
                    metadata.time_spec.get_full_secs(),
                    metadata.time_spec.get_frac_secs())

            elif metadata.error_code == uhd.types.RXMetadataErrorCode.late:
                self._dashboard_put(
                    status_queue,
                    {
                        "kind": "event",
                        "level": "WARN",
                        "message": "idx=%d late: %s" % (
                            burst_index,
                            metadata.strerror(),
                        ),
                    },
                )
                logger.warn(
                    (
                        "RX_BURST_ERR idx=%d status=late msg=%s scheduled=%.9f "
                        "issue_time=%.9f recv_done=%.9f action=restart"
                    ),
                    burst_index,
                    metadata.strerror(),
                    scheduled_rx_start,
                    host_issue_time,
                    host_recv_done,
                )
                # Radio core will be in the idle state. Issue stream command to restart streaming.
                stream_cmd.time_spec = uhd.types.TimeSpec(
                    usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY)
                stream_cmd.stream_now = True
                rx_stream.issue_stream_cmd(stream_cmd)
            elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
                self._dashboard_put(
                    status_queue,
                    {
                        "kind": "event",
                        "level": "WARN",
                        "message": "idx=%d timeout: %s" % (
                            burst_index,
                            metadata.strerror(),
                        ),
                    },
                )
                logger.warn(
                    (
                        "RX_BURST_ERR idx=%d status=timeout msg=%s scheduled=%.9f "
                        "issue_time=%.9f recv_done=%.9f"
                    ),
                    burst_index,
                    metadata.strerror(),
                    scheduled_rx_start,
                    host_issue_time,
                    host_recv_done,
                )
            else:
                self._dashboard_put(
                    status_queue,
                    {
                        "kind": "event",
                        "level": "ERR",
                        "message": "idx=%d unexpected: %s" % (
                            burst_index,
                            metadata.strerror(),
                        ),
                    },
                )
                logger.err(
                    (
                        "RX_BURST_ERR idx=%d status=unexpected msg=%s scheduled=%.9f "
                        "issue_time=%.9f recv_done=%.9f"
                    ),
                    burst_index,
                    metadata.strerror(),
                    scheduled_rx_start,
                    host_issue_time,
                    host_recv_done,
                )

        # End continuous stream.
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_stream.issue_stream_cmd(stream_cmd)
        if last_rx_event is not None:
            self._log_rx_burst(
                logger,
                last_rx_event[0],
                "last",
                last_rx_event[1],
                last_rx_event[2],
                last_rx_event[3],
                last_rx_event[4],
                last_rx_event[5],
                last_rx_event[7],
                last_rx_event[8],
                last_rx_event[9],
                last_rx_event[10],
                last_rx_event[6],
                max_samps_per_packet,
                self._queue_depth(rcv_queue),
            )
