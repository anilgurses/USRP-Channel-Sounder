from WaveformGenerator import Waveform
import math
from time import sleep

import uhd


class Transmitter:
    def __init__(self, config):
        self.config = config

    def transmit(self, usrp, tx_stream, logger, terminate):
        wv = Waveform(self.config)
        waveform = wv.create_waveform()
        tx_rate = float(usrp.get_tx_rate())
        burst_samples = int(waveform.shape[0])
        burst_duration = burst_samples / tx_rate if tx_rate > 0 else 0.0
        burst_index = 0
        last_tx_event = None
        configured_log_every = getattr(self.config, "LOG_EVERY", None)
        if configured_log_every is None:
            log_every = max(1, int(getattr(self.config, "PERIOD", 1)) * 5)
        else:
            log_every = max(1, int(configured_log_every))

        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = True
        metadata.has_time_spec = True

        logger.info(
            "TX_START rate=%.3f burst_samps=%d burst_dur=%.6f log_every=%d",
            tx_rate,
            burst_samples,
            burst_duration,
            log_every,
        )

        # Scheduling transmission
        period = self.config.PERIOD
        inc_sec = 1 / period
        start_time_tx = math.ceil(
            usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY
        )
        time_tx = start_time_tx

        while not terminate.is_set():
            burst_index += 1
            scheduled_tx_start = time_tx

            usrp_time = usrp.get_time_now().get_real_secs()
            if scheduled_tx_start <= usrp_time:
                skip_count = int((usrp_time - scheduled_tx_start) / inc_sec) + 1
                scheduled_tx_start += skip_count * inc_sec
                start_time_tx += skip_count * inc_sec
                time_tx = scheduled_tx_start
                logger.warn(
                    "TX_SCHED_SKIP idx=%d skipped=%d usrp_time=%.9f new_time_tx=%.9f"
                    " (TX timed command would have been late)",
                    burst_index, skip_count, usrp_time, time_tx,
                )

            metadata.time_spec = uhd.types.TimeSpec(scheduled_tx_start)

            host_submit_time = usrp.get_time_now().get_real_secs()
            sent_samps = tx_stream.send(waveform, metadata, 1)
            host_return_time = usrp.get_time_now().get_real_secs()

            if sent_samps > 0 and tx_rate > 0:
                scheduled_last_sample = scheduled_tx_start + (sent_samps - 1) / tx_rate
                scheduled_tx_end = scheduled_tx_start + sent_samps / tx_rate
            else:
                scheduled_last_sample = scheduled_tx_start
                scheduled_tx_end = scheduled_tx_start

            submit_lead = scheduled_tx_start - host_submit_time
            if burst_index == 1 or burst_index % log_every == 0:
                logger.info(
                    (
                        "TX_EVT idx=%d t_intended=%.9f t_last=%.9f t_end=%.9f "
                        "t_submit=%.9f t_done=%.9f lead=%.6f samps=%d"
                    ),
                    burst_index,
                    scheduled_tx_start,
                    scheduled_last_sample,
                    scheduled_tx_end,
                    host_submit_time,
                    host_return_time,
                    submit_lead,
                    sent_samps,
                )
            last_tx_event = (
                burst_index,
                scheduled_tx_start,
                scheduled_last_sample,
                scheduled_tx_end,
                host_submit_time,
                host_return_time,
                sent_samps,
            )

            usrp_time = usrp.get_time_now().get_real_secs()
            while usrp_time < time_tx:
                sleep(inc_sec / 1000)
                usrp_time = usrp.get_time_now().get_real_secs()

            # Derive the next timestamp from the initial epoch. Repeated
            # ceil-based quantization accumulates drift over long runs.
            next_time_tx = start_time_tx + burst_index * inc_sec
            # Quantize to the 0.2 ms (200 us) grid using integer ticks.
            ticks = round(next_time_tx * 1e4)
            if ticks % 2:
                ticks += 1
            time_tx = ticks / 1e4

        if last_tx_event is not None:
            logger.info(
                (
                    "TX_LAST idx=%d t_intended=%.9f t_last=%.9f t_end=%.9f "
                    "t_submit=%.9f t_done=%.9f samps=%d"
                ),
                last_tx_event[0],
                last_tx_event[1],
                last_tx_event[2],
                last_tx_event[3],
                last_tx_event[4],
                last_tx_event[5],
                last_tx_event[6],
            )
