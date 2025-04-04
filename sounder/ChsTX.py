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

        metadata = uhd.types.TXMetadata()
        metadata.time_spec = uhd.types.TimeSpec(
            usrp.get_time_now().get_full_secs() + self.config.USRP_CONF.INIT_DELAY
        )

        metadata.start_of_burst = True
        metadata.end_of_burst = True
        metadata.has_time_spec = True
        # max_samps_per_packet = tx_stream.get_max_num_samps()
        # print(max_samps_per_packet)

        num_samps = 0
        num_samps += tx_stream.send(waveform, metadata)

        logger.info("TX Stream started")

        # Scheduling transmission
        period = self.config.PERIOD
        inc_sec = 1 / period
        time_tx = usrp.get_time_now().get_real_secs() + self.config.USRP_CONF.INIT_DELAY

        while not terminate.is_set():
            usrp_time = usrp.get_time_now().get_real_secs()

            print(time_tx, usrp_time)
            metadata.time_spec = uhd.types.TimeSpec(time_tx)

            num_samps += tx_stream.send(waveform, metadata, 1)
            usrp_time = usrp.get_time_now().get_real_secs()

            while usrp_time < time_tx:
                sleep(inc_sec / 1000)
                usrp_time = usrp.get_time_now().get_real_secs()

            time_tx += inc_sec
            time_tx = math.ceil(time_tx * 1e4) / 1e4

            if (time_tx * 10 ) % 2 != 0:
                time_tx = (math.ceil((time_tx*1e4) / 2.0) * 2.0) / 1e4

            # time_tx = usrp.get_time_now().get_full_secs() + 1.0
