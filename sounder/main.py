import argparse
import signal
import tempfile
import time
import threading
from multiprocessing import Event, Manager, Process

import pyximport
import numpy as np

import uhd

pyximport.install(
    build_dir=f"{tempfile.gettempdir()}/sounder-pyxbld",
    setup_args={
        "include_dirs": [np.get_include()],
        "script_args": ["--verbose"],
    },
    language_level=3,
)

from ChsTX import Transmitter
from ChsRX import Receiver
from usrp_utils import createMultiUSRP, init_sync
from utils.config_parser import Config
from utils.logger import Logger
from utils.tui import RxDashboard

terminate_event = None


def signal_handling(_signum, _frame):
    if terminate_event is not None:
        terminate_event.set()
    raise KeyboardInterrupt


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


def stop_workers(threads, rcv_prc, logger):
    if terminate_event is not None:
        terminate_event.set()

    for thread in threads:
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warn("Worker thread %s did not stop within timeout", thread.name)

    if rcv_prc is not None and rcv_prc.pid is not None:
        rcv_prc.join(timeout=5)
        if rcv_prc.is_alive():
            logger.warn("Receiver processing process did not stop; terminating")
            rcv_prc.terminate()
            rcv_prc.join(timeout=5)


def main(args):
    global terminate_event
    terminate_event = Event()
    terminate_event.clear()
    logger = Logger(console=not args.tui)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handling)

    threads = []
    rcv_prc = None

    try:
        config = Config(args.config)
        if config.MODE not in {"TX", "RX"}:
            logger.err("Unsupported MODE '%s'. Use TX or RX.", config.MODE)
            return 1

        usrp = createMultiUSRP(config)
        logger.info("Using the Device: %s", usrp.get_pp_string())

        configure_usrp(usrp, config)
        init_sync(config, usrp, logger, terminate_event)
        usrp.clear_command_time()

        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]

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
                    daemon=True,
                )
                threads.append(tx_thread)
                tx_thread.start()
            else:
                receiver = Receiver(config, args.plot)
                rx_streamer = usrp.get_rx_stream(stream_args)
                rx_thread = threading.Thread(
                    target=receiver.receive,
                    name="rx-thread",
                    args=(
                        usrp,
                        rx_streamer,
                        logger,
                        rcv_queue,
                        pwr_queue if args.tui else None,
                        terminate_event,
                        args,
                    ),
                    daemon=True,
                )
                threads.append(rx_thread)
                rx_thread.start()

                rcv_prc = Process(
                    target=receiver.process_recv_data,
                    args=(rcv_queue, pwr_queue, logger, terminate_event, args.tui),
                )
                rcv_prc.start()

            if args.tui and config.MODE == "RX":
                dashboard = RxDashboard(
                    pwr_queue,
                    terminate_event,
                    config=config,
                )
                dashboard.run(
                    lambda: any(thread.is_alive() for thread in threads)
                    or (rcv_prc is not None and rcv_prc.is_alive())
                )
            else:
                for thread in threads:
                    thread.join()

            if rcv_prc is not None:
                rcv_prc.join()

        return 0
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        return 130
    finally:
        stop_workers(threads, rcv_prc, logger)
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USRP channel sounder runner")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting hooks where supported.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Show a live RX terminal dashboard with timing and power history.",
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
