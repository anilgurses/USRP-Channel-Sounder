import uhd
from utils.config_parser import Config
from utils.logger import Logger
from usrp_utils import createMultiUSRP, init_sync
import sys
import time
import threading
from multiprocessing import Process, Manager, Lock
import argparse
import signal
import pyximport

# import Plotter

pyximport.install(setup_args={"script_args": ["--verbose"]})

from ChsTX import Transmitter
from ChsRX import Receiver

terminate_event = threading.Event()
time_set = False
usrp = None


def signal_handling(signum, frame):
    global terminate_event
    terminate_event.set()
    sys.exit(0)


def main(args):
    global terminate_event, usrp

    # TODO make it one config file
    config = Config(args.config)

    # usrp.get_tx_power_reference()
    signal.signal(signal.SIGINT, signal_handling)

    # Only one device can be used for this script.
    if config.MODE == "TX":
        usrp = createMultiUSRP(config)
    elif config.MODE == "RX":
        usrp = createMultiUSRP(config)
    # elif config.mode == "BOTH":
    #     usrp = createMultiUSRP(config)
    else:
        sys.exit(1)

    logger = Logger()
    logger.info(f"Using the Device: {usrp.get_pp_string()}")

    usrp.set_rx_rate(config.USRP_CONF.SAMPLE_RATE, 0)
    usrp.set_tx_rate(config.USRP_CONF.SAMPLE_RATE, 0)

    usrp.clear_command_time()
    # Add delay info to yaml
    usrp.set_command_time(usrp.get_time_now() + uhd.types.TimeSpec(0.1))

    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)

    usrp.set_rx_gain(config.USRP_CONF.GAIN, 0)
    usrp.set_tx_gain(config.USRP_CONF.GAIN, 0)

    usrp.clear_command_time()

    time.sleep(0.1)  # Allowing LO to lock on the freq

    init_sync(config, usrp, logger)

    m = Manager()
    rcv_queue = m.Queue()
    pwr_queue = m.Queue()

    threads = []

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")

    st_args.channels = [0]

    lock = Lock()

    if config.MODE == "TX":
        transmitter = Transmitter(config)
        tx_streamer = usrp.get_tx_stream(st_args)
        init_sync(config, usrp, logger)
        usrp.clear_command_time()
        tx_thread = threading.Thread(
            target=transmitter.transmit,
            args=(usrp, tx_streamer, logger, terminate_event),
        )
        threads.append(tx_thread)
        tx_thread.start()

    elif config.MODE == "RX":
        receiver = Receiver(config, args.plot) 
        rx_streamer = usrp.get_rx_stream(st_args)
        init_sync(config, usrp, logger) 
        usrp.clear_command_time()
        rx_thread = threading.Thread(
            target=receiver.receive,
            args=(usrp, rx_streamer, logger, rcv_queue, terminate_event, args),
        )
        threads.append(rx_thread)
        rx_thread.start()

        rcv_prc = Process(
            target=receiver.process_recv_data,
            args=(rcv_queue, pwr_queue, logger, terminate_event),
        )
        rcv_prc.start()

    # elif args.mode == "BOTH":
    #     tx_streamer = usrp.get_tx_stream(st_args)
    #     tx_thread = threading.Thread(
    #         target=transmitter.transmit,
    #         args=(usrp, tx_streamer, logger, terminate_event),
    #     )
    #     threads.append(tx_thread)
    #     tx_thread.start()
    #
    #     rx_streamer = usrp.get_rx_stream(st_args)
    #     rx_thread = threading.Thread(
    #         target=receiver.receive,
    #         args=(usrp, rx_streamer, logger, rcv_queue, terminate_event, args),
    #     )
    #     threads.append(rx_thread)
    #     rx_thread.start()
    #
    #     # Since both transmit and receive, on the same device, 3 process created for avoiding overruns
    #     rcv_prc = Process(
    #         target=receiver.process_recv_data,
    #         args=(rcv_queue, pwr_queue, logger, terminate_event, lock),
    #     )
    #     rcv_prc_1 = Process(
    #         target=receiver.process_recv_data,
    #         args=(rcv_queue, pwr_queue, logger, terminate_event, lock),
    #     )
    #     rcv_prc_2 = Process(
    #         target=receiver.process_recv_data,
    #         args=(rcv_queue, pwr_queue, logger, terminate_event, lock),
    #     )
    #     rcv_prc.start()
    #     rcv_prc_1.start()
    #     rcv_prc_2.start()
    else:
        sys.exit(1)

    # if args.plot:
    #     plotter_proc = Process(target=Plotter.Graph, args=(pwr_queue,))
    #     plotter_proc.start()

    for thr in threads:
        thr.join()

    if config.MODE == "RX":
        rcv_prc.join()

    # if args.plot:
    #     plotter_proc.join()

    return True


if __name__ == "__main__":
    logger = Logger()

    logger.info("Started")

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--plot", type=bool, default=False, help="Plotting the metrics")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../config/rx_config.yaml",
        help="Config file",
    )

    args = parser.parse_args()

    sys.exit(not main(args))
