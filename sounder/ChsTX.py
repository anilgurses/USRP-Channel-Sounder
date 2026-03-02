import uhd
from WaveformGenerator import Waveform
import math
from time import sleep


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
        log_every = max(1, int(getattr(self.config, "PERIOD", 1)) * 5)

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
        time_tx = usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY

        while not terminate.is_set():
            burst_index += 1
            scheduled_tx_start = time_tx
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

            time_tx += inc_sec
            time_tx = math.ceil(time_tx * 1e4) / 1e4

            if (time_tx * 10) % 2 != 0:
                time_tx = (math.ceil((time_tx * 1e4) / 2.0) * 2.0) / 1e4

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
