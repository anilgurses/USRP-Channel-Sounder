import argparse
import signal
import threading
import time
from multiprocessing import Manager, Process

import pyximport
import uhd

pyximport.install(setup_args={"script_args": ["--verbose"]})

from ChsTX import Transmitter
from ChsRX import Receiver
from usrp_utils import createMultiUSRP, init_sync
from utils.config_parser import Config
from utils.logger import Logger

terminate_event = threading.Event()


def signal_handling(_signum, _frame):
    terminate_event.set()


def configure_usrp(usrp, config):
    usrp.set_rx_rate(config.USRP_CONF.SAMPLE_RATE, 0)
    usrp.set_tx_rate(config.USRP_CONF.SAMPLE_RATE, 0)

    usrp.clear_command_time()
    usrp.set_command_time(usrp.get_time_now() + uhd.types.TimeSpec(0.1))
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)
    usrp.set_rx_gain(config.USRP_CONF.GAIN, 0)
    usrp.set_tx_gain(config.USRP_CONF.GAIN, 0)
    usrp.clear_command_time()
    time.sleep(0.1)  # Allow LO to lock.


def main(args):
    terminate_event.clear()
    logger = Logger()
    signal.signal(signal.SIGINT, signal_handling)

    config = Config(args.config)
    if config.MODE not in {"TX", "RX"}:
        logger.err("Unsupported MODE '%s'. Use TX or RX.", config.MODE)
        return 1

    usrp = createMultiUSRP(config)
    logger.info("Using the Device: %s", usrp.get_pp_string())

    configure_usrp(usrp, config)
    init_sync(config, usrp, logger)
    usrp.clear_command_time()

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]

    threads = []
    rcv_prc = None

    with Manager() as manager:
        rcv_queue = manager.Queue()
        pwr_queue = manager.Queue()

        if config.MODE == "TX":
            transmitter = Transmitter(config)
            tx_streamer = usrp.get_tx_stream(stream_args)
            tx_thread = threading.Thread(
                target=transmitter.transmit,
                name="tx-thread",
                args=(usrp, tx_streamer, logger, terminate_event),
            )
            threads.append(tx_thread)
            tx_thread.start()
        else:
            receiver = Receiver(config, args.plot)
            rx_streamer = usrp.get_rx_stream(stream_args)
            rx_thread = threading.Thread(
                target=receiver.receive,
                name="rx-thread",
                args=(usrp, rx_streamer, logger, rcv_queue, terminate_event, args),
            )
            threads.append(rx_thread)
            rx_thread.start()

            rcv_prc = Process(
                target=receiver.process_recv_data,
                args=(rcv_queue, pwr_queue, logger, terminate_event),
            )
            rcv_prc.start()

        for thread in threads:
            thread.join()

        if rcv_prc is not None:
            rcv_prc.join()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USRP channel sounder runner")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting hooks where supported.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../config/rx_config.yaml",
        help="Path to YAML config file.",
    )

    parsed_args = parser.parse_args()
    raise SystemExit(main(parsed_args))
