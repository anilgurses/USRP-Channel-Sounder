"""Coordinator daemon for synchronised multi-node sounder captures.

Two run modes:

1) **Single-shot** (legacy compatible): the coordinator broadcasts one START
   message when N nodes connect, then exits. Useful for bench tests and for
   the manual A/B workflow when you don't want a full sweep.

2) **Sweep** (--sweep-tx-nodes ... --sweep-subdevs ...): the coordinator drives
   a multi-cycle state machine. Each cycle assigns one TX node and one RX
   subdev; the receivers tag their measurement directories accordingly. After
   all `expected_nodes` ack `cycle_done`, the next cycle's `cycle_start` is
   broadcast. A `manifest.json` is written incrementally to
   `<--manifest-dir>/<sweep_id>/manifest.json`.

Wire protocol (newline-delimited JSON, one message per line):

    node -> coordinator:  {"hello": "<NODE_ID>", "mode": "TX|RX|AUTO"}
    coordinator -> node:  {"command": "start", "epoch": ..., "duration": ...,
                            "channel_label": ...}                # single-shot
    coordinator -> node:  {"command": "cycle_start",
                            "cycle_index": k, "total_cycles": N,
                            "epoch": ..., "duration": ...,
                            "tx_node": "...", "rx_subdev": "A:0|B:0",
                            "channel_label": "A|B",
                            "sweep_id": "..."}                    # sweep
    node -> coordinator:  {"command": "cycle_done", "node_id": "...",
                            "cycle_index": k, "role": "TX|RX",
                            "n_frames": <int>}                    # sweep
    coordinator -> node:  {"command": "shutdown", "sweep_id": "..."} # sweep end
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class _NodeConnection:
    conn: socket.socket
    addr: tuple[str, int]
    node_id: str
    mode: str
    recv_buf: bytes = b""
    alive: bool = True


@dataclass
class _CyclePlan:
    cycle_index: int
    tx_node: str
    rx_subdev: str
    channel_label: str


def _send_json(conn: socket.socket, payload: dict) -> None:
    line = (json.dumps(payload) + "\n").encode("utf-8")
    conn.sendall(line)


def _recv_one_line(node: _NodeConnection, timeout_s: float) -> dict | None:
    deadline = time.time() + timeout_s
    while b"\n" not in node.recv_buf:
        node.conn.settimeout(max(0.05, deadline - time.time()))
        try:
            chunk = node.conn.recv(4096)
        except socket.timeout:
            return None
        except OSError:
            node.alive = False
            return None
        if not chunk:
            node.alive = False
            return None
        node.recv_buf += chunk
        if len(node.recv_buf) > 65536:
            node.alive = False
            return None
    line, _, rest = node.recv_buf.partition(b"\n")
    node.recv_buf = rest
    try:
        return json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _accept_nodes(host: str, port: int, expected: int,
                   hello_timeout_s: float) -> list[_NodeConnection]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(expected)
    print(f"[coordinator] listening on {host}:{port}, expecting {expected} node(s)")
    nodes: list[_NodeConnection] = []
    try:
        while len(nodes) < expected:
            conn, addr = server.accept()
            node = _NodeConnection(conn=conn, addr=addr, node_id="?", mode="?")
            hello = _recv_one_line(node, timeout_s=hello_timeout_s)
            if hello is None or "hello" not in hello:
                print(f"[coordinator] WARN bad hello from {addr}; dropping", file=sys.stderr)
                conn.close()
                continue
            node.node_id = str(hello.get("hello", ""))
            node.mode = str(hello.get("mode", "?"))
            nodes.append(node)
            print(f"[coordinator] connected: {node.node_id} (mode={node.mode}) "
                  f"from {addr[0]}:{addr[1]}  [{len(nodes)}/{expected}]")
    finally:
        server.close()
    return nodes


def _broadcast(nodes: list[_NodeConnection], payload: dict) -> None:
    for node in nodes:
        if not node.alive:
            continue
        try:
            _send_json(node.conn, payload)
        except OSError as exc:
            print(f"[coordinator] WARN failed to send to {node.node_id}: {exc}", file=sys.stderr)
            node.alive = False


def _build_sweep(tx_nodes: list[str], subdevs: list[str]) -> list[_CyclePlan]:
    """Build the cycle list: outer = TX rotation, inner = subdev."""
    cycles: list[_CyclePlan] = []
    idx = 0
    for tx in tx_nodes:
        for subdev in subdevs:
            label = subdev.split(":", 1)[0].strip()
            cycles.append(_CyclePlan(
                cycle_index=idx,
                tx_node=tx.strip().lower(),
                rx_subdev=subdev,
                channel_label=label.upper(),
            ))
            idx += 1
    return cycles


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    tmp.replace(path)


def run_single_shot(host: str, port: int, expected: int,
                    duration: float, lead_s: float, channel_label: str,
                    hello_timeout_s: float) -> int:
    nodes = _accept_nodes(host, port, expected, hello_timeout_s)
    epoch = math.ceil(time.time()) + lead_s
    payload = {
        "command": "start",
        "epoch": float(epoch),
        "duration": float(duration),
        "channel_label": channel_label,
    }
    print(f"[coordinator] single-shot START epoch={epoch:.3f} "
          f"(in {epoch - time.time():.2f}s), duration={duration:.1f}s, "
          f"channel={channel_label}")
    _broadcast(nodes, payload)
    for node in nodes:
        try:
            node.conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        node.conn.close()
    return 0


def run_sweep(host: str, port: int, expected: int,
              tx_nodes: list[str], subdevs: list[str],
              duration: float, lead_first_s: float, lead_per_cycle_s: float,
              cycle_ack_timeout_s: float, hello_timeout_s: float,
              manifest_dir: Path) -> int:
    sweep_id = "sweep_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    cycles = _build_sweep(tx_nodes, subdevs)
    manifest_path = manifest_dir / sweep_id / "manifest.json"
    manifest = {
        "sweep_id": sweep_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_per_cycle_s": float(duration),
        "lead_first_s": float(lead_first_s),
        "lead_per_cycle_s": float(lead_per_cycle_s),
        "expected_nodes": [],  # filled after HELLO
        "tx_nodes": list(tx_nodes),
        "subdevs": list(subdevs),
        "cycles": [],
    }

    print(f"[coordinator] sweep_id={sweep_id}  total cycles={len(cycles)}  "
          f"duration={duration:.1f}s/cycle")

    nodes = _accept_nodes(host, port, expected, hello_timeout_s)
    manifest["expected_nodes"] = [n.node_id for n in nodes]
    _write_manifest(manifest_path, manifest)
    print(f"[coordinator] manifest -> {manifest_path}")

    try:
        for i, plan in enumerate(cycles):
            alive_nodes = [n for n in nodes if n.alive]
            if not alive_nodes:
                print("[coordinator] no nodes alive; aborting sweep", file=sys.stderr)
                break
            lead = lead_first_s if i == 0 else lead_per_cycle_s
            epoch = math.ceil(time.time()) + lead
            cycle_msg = {
                "command": "cycle_start",
                "cycle_index": plan.cycle_index,
                "total_cycles": len(cycles),
                "epoch": float(epoch),
                "duration": float(duration),
                "tx_node": plan.tx_node,
                "rx_subdev": plan.rx_subdev,
                "channel_label": plan.channel_label,
                "sweep_id": sweep_id,
            }
            print(f"\n[coordinator] cycle {plan.cycle_index+1}/{len(cycles)}: "
                  f"TX={plan.tx_node}  rx_subdev={plan.rx_subdev}  "
                  f"label={plan.channel_label}  epoch={epoch:.3f} "
                  f"(in {epoch - time.time():.2f}s)")
            _broadcast(alive_nodes, cycle_msg)

            # Wait for cycle_done from every alive node, with a deadline
            # past the cycle's end
            ack_deadline = epoch + duration + cycle_ack_timeout_s
            pending = {n.node_id: n for n in alive_nodes}
            cycle_record = {
                "cycle_index": plan.cycle_index,
                "epoch": float(epoch),
                "duration": float(duration),
                "tx_node": plan.tx_node,
                "rx_subdev": plan.rx_subdev,
                "channel_label": plan.channel_label,
                "completed": [],
                "n_frames": {},
                "roles": {},
            }
            while pending and time.time() < ack_deadline:
                remaining = max(0.5, ack_deadline - time.time())
                for node in list(pending.values()):
                    msg = _recv_one_line(node, timeout_s=min(0.5, remaining))
                    if msg is None:
                        if not node.alive:
                            print(f"[coordinator] WARN {node.node_id} dropped during cycle "
                                  f"{plan.cycle_index}", file=sys.stderr)
                            pending.pop(node.node_id, None)
                        continue
                    if msg.get("command") != "cycle_done":
                        continue
                    cycle_record["completed"].append(node.node_id)
                    cycle_record["n_frames"][node.node_id] = int(msg.get("n_frames", 0))
                    cycle_record["roles"][node.node_id] = str(msg.get("role", "?"))
                    print(f"  [coordinator] ack: {node.node_id} role={msg.get('role')} "
                          f"n_frames={msg.get('n_frames', 0)}")
                    pending.pop(node.node_id, None)
            if pending:
                print(f"[coordinator] WARN cycle {plan.cycle_index} missing acks: "
                      f"{sorted(pending)}", file=sys.stderr)
            manifest["cycles"].append(cycle_record)
            _write_manifest(manifest_path, manifest)
    finally:
        # Send shutdown to anyone still connected.
        alive_nodes = [n for n in nodes if n.alive]
        _broadcast(alive_nodes, {"command": "shutdown", "sweep_id": sweep_id})
        for node in nodes:
            try:
                node.conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            try:
                node.conn.close()
            except OSError:
                pass

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\n[coordinator] sweep complete. manifest: {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address for the TCP server.")
    parser.add_argument("--port", type=int, default=5555,
                        help="TCP port for node check-ins.")
    parser.add_argument("--expected-nodes", type=int, required=True,
                        help="Number of nodes that must check in before the sweep starts.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Per-cycle (or single-shot) capture duration in seconds.")
    parser.add_argument("--lead-s", type=float, default=10.0,
                        help="Lead time before the first cycle.")
    parser.add_argument("--lead-per-cycle-s", type=float, default=5.0,
                        help="Lead time between cycles in sweep mode (gives nodes time to "
                             "reconfigure subdev/freq/gain).")
    parser.add_argument("--cycle-ack-timeout-s", type=float, default=15.0,
                        help="Grace period after the cycle's natural end to wait for "
                             "cycle_done acks before moving on.")
    parser.add_argument("--channel-label", default="A",
                        help="Channel label for single-shot mode (ignored in sweep).")
    parser.add_argument("--hello-timeout-s", type=float, default=30.0,
                        help="Max wait for a hello message on a freshly-accepted socket.")
    parser.add_argument("--sweep-tx-nodes", default=None,
                        help="Comma-separated TX rotation, e.g. 'lw1,lw2,lw3,lw4,lw5'. "
                             "Triggers sweep mode.")
    parser.add_argument("--sweep-subdevs", default="A:0,B:0",
                        help="Comma-separated subdev list (inner loop). Default A:0,B:0.")
    parser.add_argument("--manifest-dir", default=str(Path(__file__).resolve().parent.parent / "sweeps"),
                        help="Directory where the sweep manifest.json is written.")
    args = parser.parse_args(argv)

    if args.expected_nodes < 1:
        parser.error("--expected-nodes must be >= 1")
    if args.lead_s < 1.0:
        parser.error("--lead-s must be >= 1.0")

    if args.sweep_tx_nodes:
        tx_nodes = [s.strip().lower() for s in args.sweep_tx_nodes.split(",") if s.strip()]
        subdevs = [s.strip() for s in args.sweep_subdevs.split(",") if s.strip()]
        if not tx_nodes:
            parser.error("--sweep-tx-nodes must list at least one node")
        if not subdevs:
            parser.error("--sweep-subdevs must list at least one subdev spec")
        return run_sweep(
            host=args.host, port=args.port, expected=args.expected_nodes,
            tx_nodes=tx_nodes, subdevs=subdevs,
            duration=args.duration,
            lead_first_s=args.lead_s,
            lead_per_cycle_s=args.lead_per_cycle_s,
            cycle_ack_timeout_s=args.cycle_ack_timeout_s,
            hello_timeout_s=args.hello_timeout_s,
            manifest_dir=Path(args.manifest_dir),
        )

    return run_single_shot(
        host=args.host, port=args.port, expected=args.expected_nodes,
        duration=args.duration, lead_s=args.lead_s,
        channel_label=args.channel_label,
        hello_timeout_s=args.hello_timeout_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
