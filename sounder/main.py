import argparse
import json
import signal
import socket
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
    if config.MODE == "RX":
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(config.USRP_CONF.RX_SUBDEV))

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


def _resolve_node_id(config, override):
    if override:
        return str(override)
    for attr in ("TX_NODE", "RX_NODE"):
        value = getattr(config, attr, "") or config.raw.get(attr, "") if hasattr(config, "raw") else ""
        if value:
            return str(value)
    return socket.gethostname()


def _handshake_with_coordinator(addr, node_id, mode, logger, timeout_s=600.0):
    """Block until the coordinator broadcasts START. Returns
    (start_epoch, duration, channel_label)."""
    host, _, port = addr.partition(":")
    if not host or not port:
        raise ValueError(f"--coordinator must be HOST:PORT, got {addr!r}")
    port = int(port)
    logger.info("connecting to coordinator %s:%d as %s/%s", host, port, node_id, mode)
    sock = socket.create_connection((host, port), timeout=30.0)
    sock.settimeout(timeout_s)
    try:
        hello = json.dumps({"hello": node_id, "mode": mode}) + "\n"
        sock.sendall(hello.encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("coordinator closed connection before sending START")
            buf += chunk
            if len(buf) > 65536:
                raise RuntimeError("coordinator sent oversized message")
        line, _ = buf.split(b"\n", 1)
        msg = json.loads(line.decode("utf-8"))
    finally:
        try:
            sock.close()
        except OSError:
            pass
    if msg.get("command") != "start":
        raise RuntimeError(f"unexpected coordinator message: {msg!r}")
    start_epoch = float(msg["epoch"])
    duration = float(msg["duration"])
    channel_label = str(msg.get("channel_label", "A"))
    logger.info("coordinator START: epoch=%.6f duration=%.3f channel_label=%s",
                start_epoch, duration, channel_label)
    return start_epoch, duration, channel_label


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

        node_id = _resolve_node_id(config, getattr(args, "node_id", None))
        start_epoch = args.start_epoch
        duration = args.duration
        channel_label = (args.rx_channel_label
                         if args.rx_channel_label is not None
                         else config.USRP_CONF.RX_CHANNEL_LABEL)
        if args.coordinator:
            start_epoch, duration, channel_label = _handshake_with_coordinator(
                args.coordinator, node_id, config.MODE, logger,
            )

        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]

        with Manager() as manager:
            rcv_queue = manager.Queue()
            pwr_queue = manager.Queue()

            if config.MODE == "TX":
                transmitter = Transmitter(config, start_epoch=start_epoch, duration=duration)
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
                receiver = Receiver(
                    config, args.plot,
                    start_epoch=start_epoch, duration=duration,
                    channel_label=channel_label,
                )
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
    parser.add_argument(
        "--coordinator", type=str, default=None, metavar="HOST:PORT",
        help="Connect to a coordinator daemon; wait for START before capturing.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Capture duration in seconds. Required for bounded captures; "
             "ignored if --coordinator is set (the coordinator supplies it).",
    )
    parser.add_argument(
        "--start-epoch", type=float, default=None,
        help="UTC start epoch (float seconds). Manual override, bypasses coordinator.",
    )
    parser.add_argument(
        "--rx-channel-label", type=str, default=None,
        help="Label for the active RX daughterboard (A or B); used to tag the "
             "measurement directory and per-burst metadata. Overrides config.",
    )
    parser.add_argument(
        "--node-id", type=str, default=None,
        help="Node identifier sent to the coordinator. Defaults to "
             "config.TX_NODE/RX_NODE, falling back to hostname.",
    )

    parsed_args = parser.parse_args()
    raise SystemExit(main(parsed_args))
