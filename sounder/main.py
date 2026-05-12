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
from usrp_utils import createMultiUSRP, init_sync, configure_role
from utils.config_parser import Config
from utils.logger import Logger
from utils.tui import RxDashboard

terminate_event = None


class CoordinatorClient:
    def __init__(self, addr, node_id, mode, logger, timeout_s=None):
        host, _, port = addr.partition(":")
        if not host or not port:
            raise ValueError(f"--coordinator must be HOST:PORT, got {addr!r}")
        self.node_id = node_id
        self.mode = mode
        self.logger = logger
        self.sock = socket.create_connection((host, int(port)), timeout=30.0)
        if timeout_s is not None:
            self.sock.settimeout(timeout_s)
        self._buf = b""

    def send_hello(self):
        msg = json.dumps({"hello": self.node_id, "mode": self.mode}) + "\n"
        self.sock.sendall(msg.encode("utf-8"))

    def send_cycle_done(self, cycle_index, role, n_frames):
        payload = {
            "command": "cycle_done",
            "node_id": self.node_id,
            "cycle_index": int(cycle_index),
            "role": role,
            "n_frames": int(n_frames),
        }
        try:
            self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError as exc:
            self.logger.warn("could not send cycle_done: %s", exc)

    def recv_message(self):
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            self._buf += chunk
            if len(self._buf) > 65536:
                raise RuntimeError("coordinator sent oversized message")
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return json.loads(line.decode("utf-8"))

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

def signal_handling(_signum, _frame):
    if terminate_event is not None:
        terminate_event.set()
    raise KeyboardInterrupt


def _resolve_node_id(config, override):
    if override:
        return str(override)
    for attr in ("TX_NODE", "RX_NODE"):
        value = getattr(config, attr, "") or config.raw.get(attr, "") if hasattr(config, "raw") else ""
        if value:
            return str(value)
    return socket.gethostname()

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


def _run_rx_cycle(usrp, config, logger, args, manager,
                  start_epoch, duration, channel_label,
                  rx_subdev=None, tx_node=None, out_subdir=None):
    global terminate_event
    terminate_event.clear()

    configure_role(usrp, config, "RX", rx_subdev=rx_subdev)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(stream_args)

    rcv_queue = manager.Queue()
    pwr_queue = manager.Queue()

    receiver = Receiver(
        config, args.plot,
        start_epoch=start_epoch, duration=duration,
        channel_label=channel_label,
        out_subdir=out_subdir,
        tx_node=tx_node,
    )
    rx_thread = threading.Thread(
        target=receiver.receive,
        name="rx-thread",
        args=(usrp, rx_streamer, logger, rcv_queue,
              pwr_queue if args.tui else None,
              terminate_event, args),
        daemon=True,
    )
    rcv_prc = Process(
        target=receiver.process_recv_data,
        args=(rcv_queue, pwr_queue, logger, terminate_event, args.tui),
    )
    rx_thread.start()
    rcv_prc.start()

    if args.tui:
        dashboard = RxDashboard(pwr_queue, terminate_event, config=config)
        dashboard.run(lambda: rx_thread.is_alive() or rcv_prc.is_alive())
    else:
        rx_thread.join()

    terminate_event.set()
    rcv_prc.join(timeout=10)
    if rcv_prc.is_alive():
        logger.warn("rcv_prc did not exit; terminating")
        rcv_prc.terminate()
        rcv_prc.join(timeout=5)

    return receiver


def _run_tx_cycle(usrp, config, logger, start_epoch, duration, tx_subdev=None):
    global terminate_event
    terminate_event.clear()

    configure_role(usrp, config, "TX", tx_subdev=tx_subdev)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(stream_args)

    transmitter = Transmitter(config, start_epoch=start_epoch, duration=duration)
    tx_thread = threading.Thread(
        target=transmitter.transmit,
        name="tx-thread",
        args=(usrp, tx_streamer, logger, terminate_event),
        daemon=True,
    )
    tx_thread.start()
    tx_thread.join()
    return transmitter


def _standalone_run(usrp, config, logger, args, manager, start_epoch, duration, channel_label):
    """Standalone path. Used when --coordinator is unset,
    or when the coordinator is in single-shot mode."""
    if config.MODE == "TX":
        configure_role(usrp, config, "TX")
        _run_tx_cycle(usrp, config, logger,
                      start_epoch=start_epoch, duration=duration)
    elif config.MODE == "RX":
        _run_rx_cycle(usrp, config, logger, args, manager,
                      start_epoch=start_epoch, duration=duration,
                      channel_label=channel_label)
    else:
        raise RuntimeError(f"Unsupported MODE {config.MODE!r}; use TX or RX")


def _sweep_loop(usrp, config, logger, args, manager, coord, node_id):
    while True:
        msg = coord.recv_message()
        if msg is None:
            logger.info("coordinator connection closed")
            break
        cmd = msg.get("command")
        if cmd == "shutdown":
            logger.info("coordinator shutdown received")
            break
        if cmd != "cycle_start":
            logger.warn("unexpected coordinator message: %r", msg)
            continue

        cycle_index = int(msg["cycle_index"])
        total = int(msg.get("total_cycles", 0))
        start_epoch = float(msg["epoch"])
        duration = float(msg["duration"])
        tx_node = str(msg["tx_node"]).lower()
        rx_subdev = str(msg["rx_subdev"])
        channel_label = str(msg.get("channel_label", "A"))
        sweep_id = str(msg.get("sweep_id", "sweep"))

        is_tx = (node_id.lower() == tx_node)
        role = "TX" if is_tx else "RX"
        logger.info(
            "CYCLE_START idx=%d/%d role=%s tx=%s rx_subdev=%s ch=%s epoch=%.3f dur=%.1f",
            cycle_index + 1, total, role, tx_node, rx_subdev, channel_label,
            start_epoch, duration,
        )

        try:
            if is_tx:
                runner = _run_tx_cycle(
                    usrp, config, logger,
                    start_epoch=start_epoch, duration=duration,
                )
                n_frames = int(getattr(runner, "frame_count", 0))
            else:
                out_subdir = f"{sweep_id}/cycle_{cycle_index:02d}_tx{tx_node}_ch{channel_label}"
                runner = _run_rx_cycle(
                    usrp, config, logger, args, manager,
                    start_epoch=start_epoch, duration=duration,
                    channel_label=channel_label,
                    rx_subdev=rx_subdev,
                    tx_node=tx_node,
                    out_subdir=out_subdir,
                )
                n_frames = int(getattr(runner, "frame_count", 0))
        except Exception as exc:
            logger.err("cycle %d failed: %s", cycle_index, exc)
            n_frames = 0

        coord.send_cycle_done(cycle_index, role, n_frames)
        logger.info("CYCLE_DONE idx=%d role=%s n_frames=%d", cycle_index, role, n_frames)


def main(args):
    global terminate_event
    terminate_event = Event()
    terminate_event.clear()
    logger = Logger(console=not args.tui)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handling)

    try:
        config = Config(args.config)

        usrp = createMultiUSRP(config)
        logger.info("Using the Device: %s", usrp.get_pp_string())

        init_sync(config, usrp, logger, terminate_event)
        usrp.clear_command_time()

        node_id = _resolve_node_id(config, getattr(args, "node_id", None))

        with Manager() as manager:
            if args.coordinator:
                coord = CoordinatorClient(args.coordinator, node_id, config.MODE, logger)
                coord.send_hello()
                first = coord.recv_message()
                if first is None:
                    logger.err("coordinator closed before sending any command")
                    return 1
                if first.get("command") == "start":
                    start_epoch = float(first["epoch"])
                    duration = float(first["duration"])
                    channel_label = str(first.get("channel_label", "A"))
                    coord.close()
                    _standalone_run(usrp, config, logger, args, manager,
                                start_epoch, duration, channel_label)
                elif first.get("command") == "cycle_start":
                    coord._buf = (json.dumps(first) + "\n").encode("utf-8") + coord._buf
                    _sweep_loop(usrp, config, logger, args, manager, coord, node_id)
                    coord.close()
                else:
                    logger.err("unexpected first message from coordinator: %r", first)
                    coord.close()
                    return 1
            else:
                # No coordinator: --start-epoch / --duration / run-til-SIGINT.
                if config.MODE not in {"TX", "RX"}:
                    logger.err("Unsupported MODE '%s'; use TX or RX or supply --coordinator.",
                               config.MODE)
                    return 1
                channel_label = (args.rx_channel_label
                                 if args.rx_channel_label is not None
                                 else config.USRP_CONF.RX_CHANNEL_LABEL)
                _standalone_run(usrp, config, logger, args, manager,
                            args.start_epoch, args.duration, channel_label)

        return 0
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        return 130
    finally:
        if terminate_event is not None:
            terminate_event.set()
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
        help="Connect to a coordinator daemon. Single-shot if it sends START; "
             "multi-cycle sweep if it sends cycle_start.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Capture duration in seconds. Manual mode only; ignored when "
             "--coordinator is set (coordinator supplies it per cycle).",
    )
    parser.add_argument(
        "--start-epoch", type=float, default=None,
        help="UTC start epoch (float seconds). Manual override; bypasses coordinator.",
    )
    parser.add_argument(
        "--rx-channel-label", type=str, default=None,
        help="Label for the active RX daughterboard (A or B). Manual mode only; "
             "in sweep mode the coordinator supplies this per cycle.",
    )
    parser.add_argument(
        "--node-id", type=str, default=None,
        help="Node identifier sent to the coordinator. Defaults to "
             "config.TX_NODE/RX_NODE, falling back to hostname.",
    )

    parsed_args = parser.parse_args()
    raise SystemExit(main(parsed_args))
