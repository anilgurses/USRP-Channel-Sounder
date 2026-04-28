import math
import time
from collections import deque
from queue import Empty


def _fmt(value, suffix="", precision=2):
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return "n/a"
        return f"{value:.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _sparkline(values, width=112, height=10):
    points = [v for v in values if v is not None and math.isfinite(v)]
    if not points:
        return "waiting for power samples"

    points = points[-width:]
    min_v = min(points)
    max_v = max(points)
    if max_v == min_v:
        max_v += 1.0
        min_v -= 1.0

    span = max_v - min_v
    rows = []
    for row in range(height):
        threshold = max_v - (row / max(1, height - 1)) * span
        rows.append("".join("█" if value >= threshold else " " for value in points))
    return "\n".join(rows)


def _avg(values):
    points = [v for v in values if v is not None and math.isfinite(v)]
    if not points:
        return None
    return sum(points) / len(points)


class RxDashboard:
    def __init__(self, queue, terminate, refresh_hz=4, history_len=240, config=None):
        self.queue = queue
        self.terminate = terminate
        self.refresh_s = 1.0 / refresh_hz
        self.config = config
        self.power_history = deque(maxlen=history_len)
        self.burst = {}
        self.metrics = {}
        self.events = deque(maxlen=8)
        self.started = time.time()

        try:
            from rich import box
            from rich.align import Align
            from rich.console import Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as ex:
            raise RuntimeError(
                "TUI mode requires the 'rich' package. Install requirements.txt first."
            ) from ex

        self.box = box
        self.Align = Align
        self.Group = Group
        self.Live = Live
        self.Panel = Panel
        self.Table = Table
        self.Text = Text

    def _drain(self):
        drained = 0
        while True:
            try:
                item = self.queue.get(False)
            except Empty:
                break
            except Exception:
                break

            drained += 1
            kind = item.get("kind")
            if kind == "rx_burst":
                self.burst = item
            elif kind == "rx_metrics":
                self.metrics = item
                power = item.get("power_dbm")
                if power is not None and math.isfinite(power):
                    self.power_history.append(power)
            elif kind == "event":
                self.events.append(item)
        return drained

    def _plain_table(self):
        table = self.Table.grid(expand=True)
        table.add_column("label", style="dim", no_wrap=True, ratio=1)
        table.add_column("value", justify="right", ratio=1)
        return table

    def _status_style(self, value):
        if value in {"ok", "DETECTED", "OK", "no"}:
            return "green"
        if value in {"short", "SATURATED", "yes"}:
            return "yellow"
        if value in {"MISSED", "NO SIGNAL"}:
            return "red"
        if value in {"overflow", "late", "timeout", "unexpected"}:
            return "red"
        return "white"

    def _styled(self, value):
        text = str(value)
        return self.Text(text, style=self._status_style(text))

    def _status_panel(self):
        table = self._plain_table()

        elapsed = time.time() - self.started
        table.add_row("frame", str(self.metrics.get("idx", self.burst.get("idx", "n/a"))))
        table.add_row("status", self._styled(self.burst.get("status", "waiting")))
        table.add_row("samples", self.burst.get("sample_text", "n/a"))
        table.add_row("missing", str(self.burst.get("missing_samps", "n/a")))
        table.add_row("queue", str(self.burst.get("queue_depth", "n/a")))
        table.add_row("elapsed", _fmt(elapsed, "s", 1))
        return self.Panel(table, title="Receiver", border_style="cyan", box=self.box.ROUNDED)

    def _config_panel(self):
        table = self.Table.grid(expand=True)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_column("spacer", width=3)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")

        config = self.config
        if config is None:
            table.add_row("config", "n/a", "", "", "")
            return self.Panel(table, title="Config", border_style="white", box=self.box.ROUNDED)

        table.add_row(
            "mode",
            str(getattr(config, "MODE", "n/a")),
            "",
            "serial",
            str(getattr(config.USRP_CONF, "SERIAL", "n/a")),
        )
        table.add_row(
            "center",
            _fmt(float(getattr(config.USRP_CONF, "CENTER_FREQ", 0.0)) / 1e6, " MHz", 3),
            "",
            "rate",
            _fmt(float(getattr(config.USRP_CONF, "SAMPLE_RATE", 0.0)) / 1e6, " MS/s", 3),
        )
        table.add_row(
            "gain",
            _fmt(getattr(config.USRP_CONF, "GAIN", None), " dB", 1),
            "",
            "period",
            _fmt(getattr(config, "PERIOD", None), " Hz", 2),
        )
        table.add_row(
            "duration",
            _fmt(getattr(config.RX, "DURATION", None), " s", 4),
            "",
            "waveform",
            str(getattr(config, "WAVEFORM", "n/a")),
        )
        table.add_row(
            "clock/PPS",
            "%s/%s" % (
                getattr(config.USRP_CONF, "CLK_REF", "n/a"),
                getattr(config.USRP_CONF, "PPS_REF", "n/a"),
            ),
            "",
            "output",
            str(getattr(config.RX, "OUTPUT_TYPE", "n/a")),
        )
        return self.Panel(table, title="Config", border_style="white", box=self.box.ROUNDED)

    def _timing_panel(self):
        table = self.Table.grid(expand=True)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_column("spacer", width=3)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")

        table.add_row(
            "schedule error",
            _fmt(self.burst.get("schedule_error_us"), " us", 3),
            "",
            "issue lead",
            _fmt(self.burst.get("issue_lead_ms"), " ms", 3),
        )
        table.add_row(
            "recv wait",
            _fmt(self.burst.get("recv_wait_ms"), " ms", 3),
            "",
            "rx duration",
            _fmt(self.burst.get("rx_duration_ms"), " ms", 3),
        )
        table.add_row(
            "last PPS",
            _fmt(self.burst.get("last_pps"), "", 6),
            "",
            "RX-PPS offset",
            _fmt(self.burst.get("rx_pps_offset"), " s", 6),
        )
        table.add_row(
            "PPS drift",
            _fmt(self.burst.get("pps_drift_us"), " us", 3),
            "",
            "PPS stale",
            self._styled(self.burst.get("pps_stale", "n/a")),
        )
        return self.Panel(table, title="Timing", border_style="blue", box=self.box.ROUNDED)

    def _measurements_panel(self):
        table = self.Table.grid(expand=True)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_column("spacer", width=3)
        table.add_column("metric", style="dim", no_wrap=True)
        table.add_column("value", justify="right")

        table.add_row(
            "power",
            _fmt(self.metrics.get("power_dbm"), " dBm", 2),
            "",
            "mean",
            _fmt(self.metrics.get("power_dbfs"), " dBFS", 2),
        )
        table.add_row(
            "peak",
            _fmt(self.metrics.get("peak_dbfs"), " dBFS", 2),
            "",
            "crest",
            _fmt(self.metrics.get("crest_db"), " dB", 2),
        )
        table.add_row(
            "detect",
            self._styled(self.metrics.get("detect_status", "n/a")),
            "",
            "corr SNR",
            _fmt(self.metrics.get("corr_snr_db"), " dB", 1),
        )
        table.add_row(
            "RSRP",
            _fmt(self.metrics.get("rsrp_dbm"), " dBm", 2),
            "",
            "OFDM SNR",
            _fmt(self.metrics.get("ofdm_snr_db"), " dB", 2),
        )
        table.add_row(
            "path loss",
            _fmt(self.metrics.get("pl_db"), " dB", 2),
            "",
            "match",
            _fmt(self.metrics.get("norm_peak"), "", 2),
        )
        table.add_row(
            "",
            "",
            "",
            "saturation",
            self._styled(self.metrics.get("sat_status", "n/a")),
        )
        return self.Panel(table, title="Signal", border_style="green", box=self.box.ROUNDED)

    def _plot_panel(self):
        history = list(self.power_history)
        line = _sparkline(history)
        latest = history[-1] if history else None
        lo = min(history) if history else None
        hi = max(history) if history else None
        avg = _avg(history)

        stats = self.Text()
        stats.append("now ", style="dim")
        stats.append(_fmt(latest, " dBm", 2), style="bold")
        stats.append("   avg ", style="dim")
        stats.append(_fmt(avg, " dBm", 2))
        stats.append("   min ", style="dim")
        stats.append(_fmt(lo, " dBm", 2))
        stats.append("   max ", style="dim")
        stats.append(_fmt(hi, " dBm", 2))

        body = self.Group(stats, self.Text(line, style="cyan"))
        return self.Panel(body, title="Power History", border_style="magenta", box=self.box.ROUNDED)

    def _events_panel(self):
        if not self.events:
            content = self.Text("no warnings", style="dim")
        else:
            content = self.Text()
            for event in self.events:
                level = event.get("level", "INFO")
                style = "red" if level == "ERR" else "yellow"
                content.append(level, style=style)
                content.append(f" {event.get('message', '')}\n")
        return self.Panel(content, title="Events", border_style="yellow", box=self.box.ROUNDED)

    def render(self):
        title = self.Text("USRP Channel Sounder RX", style="bold cyan")
        summary = self.Table.grid(expand=True)
        summary.add_column(ratio=1)
        summary.add_column(width=1)
        summary.add_column(ratio=2)
        summary.add_row(self._status_panel(), "", self._measurements_panel())
        return self.Group(
            self.Align.center(title),
            summary,
            self._config_panel(),
            self._timing_panel(),
            self._plot_panel(),
            self._events_panel(),
        )

    def run(self, workers_alive):
        with self.Live(self.render(), refresh_per_second=1 / self.refresh_s, screen=True) as live:
            while not self.terminate.is_set() and workers_alive():
                self._drain()
                live.update(self.render())
                time.sleep(self.refresh_s)
            self._drain()
            live.update(self.render())
