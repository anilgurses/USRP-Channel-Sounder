cimport cython
import uhd

from scipy.signal import correlate
import time
from time import sleep
from scipy.io import savemat
from queue import Empty
from BufferToProcess import BufferToProcess
from WaveformGenerator import Waveform
from datetime import datetime
import os
import math
# from dronekit import connect
from ublox_gps import UbloxGps
import yaml
import serial

import numpy as np
cimport numpy as np

np.import_array()

# ctypedef Config ConfigType

class Receiver:
    # cdef ConfigType config
    # cdef int plot

    def __init__(self, config, plot):
        self.config = config
        self.plot = plot
        self.output_type = self.config.RX.OUTPUT_TYPE


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
        # RMS of signal
        # power = 10*np.log10(np.abs(np.sqrt(np.mean(sig ** 2))))
        # Ergodic Signal

        # dBFS
        power = 10 * np.log10(np.abs(np.var(sig[:self.calc_window])))
        # dBm
        power += self.config.CAL.RX_REF

        logger.info(f"Power: {power} dBm")

        return power

    def process_recv_data(self, que, pwr_queue, logger, terminate):
        # cdef int modOrder = 2
        # cdef float maxToneOffset = self.config.MAX_FREQ_OFF * modOrder
        # cdef int numLag = int(np.round(self.config.USRP_CONF.SAMPLE_RATE/maxToneOffset)) - 1
        # cuFreqOff = 0

        wv = Waveform(self.config)
        waveform = wv.create_waveform()
        self.calc_window = waveform.shape[0] * 2

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

            # freq_off, raisedBuff = coarse_freq_est(buff, numLag, raisedBuff)
            # freq_vec = freq_off * (1/SAMPLE_RATE) * np.linspace(0, 1, buff.size)
            # cuFreqOff -= freq_vec[buff.size-1]
            # print(cuFreqOff)
            # pr_id, prb_filter_states = self.preamble_detector(buff, prb_intr, prb_filter_states, thresh)
            # if len(pr_id) > 1:
            #     frameEnd = int(crp.size * self.config.CHIRP_DURATION) + 26 + pr_id[0]
            #     buff = buff[0,pr_id[0]+28:]
            #     # print(pr_id[0])
            # else :
            #     continue

            if self.config.RX.POWER_CALC:
                power = self.calc_power(buff, logger)

                if self.config.RX.PL_CALC:
                    pl = self.config.CAL.TX_REF / power
                    logger.info(f"Pathloss: {pl} dBm")

            if self.config.RX.OUTPUT_TYPE == "npz":
                np.savez(f"{out_dir}/received_{rcv_time}.npz",
                         rcv    =   buff,
                         ref   =  waveform,
                         rx_time  = rcv_time,
                         gps_info   =  gps_info,
                         time_info  =  time_info,
                         pps_info   =  pps_info,
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
                            "pps_info": pps_info
                        })


    def receive(self, usrp, rx_stream, logger, rcv_queue, terminate, args):
        gps = None
        gps_t = None

        if self.config.GPS.ENABLED:
            if self.config.GPS.SOURCE == "gnss":
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
        log_every = max(1, int(getattr(self.config, "PERIOD", 1)) * 5)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        time.sleep(1)
        logger.info(
            "RX_START rate=%.3f req_samps=%d log_every=%d",
            rate,
            max_samps_per_packet,
            log_every,
        )

        time_rx = math.ceil(usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY)

        period = self.config.PERIOD
        inc_sec = 1 / period

        while not terminate.is_set():
            try:
                burst_index += 1
                scheduled_rx_start = time_rx
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
                pps_info = {
                    "last_pps_usrp_time": last_pps_usrp_time,
                    "rx_start_minus_last_pps": actual_rx_start - last_pps_usrp_time,
                    "frame_rx_start_usrp_time": actual_rx_start,
                }

            except RuntimeError as ex:
                logger.err("Runtime error in receive: %s", ex)
                return

            usrp_time = usrp.get_time_now().get_real_secs()
            time_rx += inc_sec

            # Avoid overflow
            if time_rx <= usrp_time:
                mlt = (usrp_time - time_rx) / inc_sec + 1
                time_rx += mlt * inc_sec

            time_rx = math.ceil(time_rx * 1e4) / 1e4
             
            if (time_rx * 10 ) % 2 != 0:
                time_rx = (math.ceil((time_rx*1e4) / 2.0) * 2.0) / 1e4

            if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                schedule_error = actual_rx_start - scheduled_rx_start
                if burst_index == 1 or burst_index % log_every == 0:
                    logger.info(
                        (
                            "RX_EVT idx=%d t_intended=%.9f t_rx=%.9f t_last=%.9f "
                            "t_issue=%.9f t_done=%.9f sched_err=%.6f "
                            "t_last_pps=%.9f rx_pps_off=%.9f samps=%d"
                        ),
                        burst_index,
                        scheduled_rx_start,
                        actual_rx_start,
                        actual_last_sample,
                        host_issue_time,
                        host_recv_done,
                        schedule_error,
                        pps_info["last_pps_usrp_time"],
                        pps_info["rx_start_minus_last_pps"],
                        num_rx_samps,
                    )
                last_rx_event = (
                    burst_index,
                    scheduled_rx_start,
                    actual_rx_start,
                    actual_last_sample,
                    host_issue_time,
                    host_recv_done,
                    num_rx_samps,
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
                        recv_buffer.copy(),
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
                logger.warn(
                    (
                        "RX_ERR idx=%d code=overflow t_intended=%.9f "
                        "t_issue=%.9f t_done=%.9f"
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
                logger.warn(
                    (
                        "RX_ERR idx=%d code=late msg=%s t_intended=%.9f "
                        "t_issue=%.9f t_done=%.9f action=restart"
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
                logger.warn(
                    (
                        "RX_ERR idx=%d code=timeout msg=%s t_intended=%.9f "
                        "t_issue=%.9f t_done=%.9f"
                    ),
                    burst_index,
                    metadata.strerror(),
                    scheduled_rx_start,
                    host_issue_time,
                    host_recv_done,
                )
            else:
                logger.err(
                    (
                        "RX_ERR idx=%d code=unexpected msg=%s t_intended=%.9f "
                        "t_issue=%.9f t_done=%.9f"
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
            logger.info(
                (
                    "RX_LAST idx=%d t_intended=%.9f t_rx=%.9f t_last=%.9f "
                    "t_issue=%.9f t_done=%.9f samps=%d"
                ),
                last_rx_event[0],
                last_rx_event[1],
                last_rx_event[2],
                last_rx_event[3],
                last_rx_event[4],
                last_rx_event[5],
                last_rx_event[6],
            )
