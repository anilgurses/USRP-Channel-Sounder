"""Coordinator daemon for synchronised multi-node sounder experiments

Each node connects over TCP, identifies itself, and waits for a
START message that contains an absolute UTC start epoch, the capture duration,
and a channel label. When the configured number of nodes have connected, the
coordinator picks T = ceil(now()) + lead_s and broadcasts the START to all
connections at once. 

    node -> coordinator:  {"hello": "<NODE_ID>", "mode": "TX"|"RX"}
    coordinator -> node:  {"command": "start",
                            "epoch": <UTC seconds, float>,
                            "duration": <seconds, float>,
                            "channel_label": "A"|"B"}

Usage:
    # control host
    python3 coordinator.py --expected-nodes 5 --duration 30 --channel-label A

    # each node 
    python3 main.py --coordinator <CTRL_HOST>:5555 -c ../config/f2f/longseq_56mhz_rx.yaml

"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class _NodeConnection:
    conn: socket.socket
    addr: tuple[str, int]
    node_id: str
    mode: str


def _read_hello(conn: socket.socket, timeout_s: float) -> dict | None:
    conn.settimeout(timeout_s)
    buf = b""
    while b"\n" not in buf:
        try:
            chunk = conn.recv(1024)
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf += chunk
        if len(buf) > 4096:
            return None
    line, _ = buf.split(b"\n", 1)
    try:
        return json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _broadcast_start(nodes: list[_NodeConnection], payload: dict) -> None:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    for node in nodes:
        try:
            node.conn.sendall(encoded)
            node.conn.shutdown(socket.SHUT_WR)
        except OSError as exc:
            print(f"[coordinator] WARN failed to send START to {node.node_id}: {exc}",
                  file=sys.stderr)
        finally:
            try:
                node.conn.close()
            except OSError:
                pass


def run(host: str, port: int, expected: int, duration: float,
        lead_s: float, channel_label: str, hello_timeout_s: float) -> int:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(expected)
    print(f"[coordinator] listening on {host}:{port}, expecting {expected} node(s)")
    print(f"[coordinator] duration={duration:.1f}s lead={lead_s:.1f}s channel={channel_label}")

    nodes: list[_NodeConnection] = []
    lock = threading.Lock()
    all_connected = threading.Event()

    def _accept_loop() -> None:
        while not all_connected.is_set():
            try:
                conn, addr = server.accept()
            except OSError:
                return
            hello = _read_hello(conn, hello_timeout_s)
            if hello is None or "hello" not in hello:
                print(f"[coordinator] WARN bad msg from {addr}; dropping", file=sys.stderr)
                conn.close()
                continue
            node_id = str(hello.get("hello", ""))
            mode = str(hello.get("mode", "?"))
            with lock:
                nodes.append(_NodeConnection(conn, addr, node_id, mode))
                count = len(nodes)
                print(f"[coordinator] connected: {node_id} (mode={mode}) "
                      f"from {addr[0]}:{addr[1]}  [{count}/{expected}]")
                if count >= expected:
                    all_connected.set()

    accept_thread = threading.Thread(target=_accept_loop, daemon=True)
    accept_thread.start()
    all_connected.wait()

    try:
        server.close()
    except OSError:
        pass

    epoch = math.ceil(time.time()) + lead_s
    payload = {
        "command": "start",
        "epoch": float(epoch),
        "duration": float(duration),
        "channel_label": channel_label,
    }
    print(f"[coordinator] broadcasting START: epoch={epoch:.3f} (in "
          f"{epoch - time.time():.2f}s), duration={duration:.1f}s, channel={channel_label}")
    _broadcast_start(nodes, payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address for the TCP server.")
    parser.add_argument("--port", type=int, default=5555,
                        help="TCP port for node check-ins.")
    parser.add_argument("--expected-nodes", type=int, required=True,
                        help="Number of nodes that must check in before broadcasting START.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Capture duration in seconds (sent to each node).")
    parser.add_argument("--lead-s", type=float, default=10.0,
                        help="Seconds between full-quorum and the broadcast UTC epoch.")
    parser.add_argument("--channel-label", default="A",
                        help="Channel label echoed to all nodes for RX output tagging.")
    parser.add_argument("--hello-timeout-s", type=float, default=30.0,
                        help="Max wait for a hello message on a freshly-accepted socket.")
    args = parser.parse_args(argv)

    if args.expected_nodes < 1:
        parser.error("--expected-nodes must be >= 1")
    if args.lead_s < 1.0:
        parser.error("--lead-s must be >= 1.0 to give clients time to arm timed streams")

    return run(
        host=args.host,
        port=args.port,
        expected=args.expected_nodes,
        duration=args.duration,
        lead_s=args.lead_s,
        channel_label=args.channel_label,
        hello_timeout_s=args.hello_timeout_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
