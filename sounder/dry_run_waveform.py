import argparse
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import signal

from WaveformGenerator import Waveform
from utils.config_parser import Config

C_LIGHT = 299_792_458.0  # m/s


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_waveform_sections(cfg, wv: Waveform):
    if cfg.WAVEFORM not in ("PN", "ZC", "CHIRP"):
        raise ValueError(f"Unsupported waveform type: {cfg.WAVEFORM}")
    return wv.create_waveform(return_sections=True)


def _detection_reference(cfg, wv: Waveform):
    if cfg.WAVEFORM == "ZC":
        return wv.create_zadoff_chu()
    return None


def _format_params(cfg, total_samples: int) -> list[str]:
    fs = float(cfg.USRP_CONF.SAMPLE_RATE)
    bw = float(cfg.FILTER.BW) if cfg.FILTER.ENABLED and cfg.FILTER.BW > 0 else fs
    fc = float(cfg.USRP_CONF.CENTER_FREQ)
    period_hz = float(cfg.PERIOD)

    dt_ns = 1e9 / fs if fs > 0 else float("nan")
    range_res_m = C_LIGHT / bw if bw > 0 else float("nan")
    burst_dur_us = total_samples / fs * 1e6 if fs > 0 else 0.0
    burst_period_ms = (1.0 / period_hz) * 1e3 if period_hz > 0 else float("inf")

    lines = [
        "RF / Sampling",
        f"  Center freq           : {fc/1e9:.4f} GHz",
        f"  Sample rate (Fs)      : {fs/1e6:.3f} MHz",
        f"  Effective bandwidth   : {bw/1e6:.3f} MHz"
        + ("  (filtered)" if cfg.FILTER.ENABLED else ""),
        "",
        "Resolution",
        f"  Sample period (1/Fs)  : {dt_ns:.2f} ns",
        f"  Range resolution      : {range_res_m:.3f} m  (one-way, c/BW)",
        "",
        "Burst Timing",
        f"  Burst duration        : {burst_dur_us:.2f} us  ({total_samples} samples)",
        f"  Burst repetition      : {burst_period_ms:.2f} ms  (PERIOD = {period_hz:g} Hz)",
    ]

    if cfg.WAVEFORM == "ZC":
        seq_len = int(cfg.WAV_OPTS.SEQ_LEN)
        n_rep = int(getattr(cfg.WAV_OPTS, "ZC_NUM_REPEATS", 4))
        guard = int(getattr(cfg.WAV_OPTS, "GUARD_LEN_SAMPS", 100))
        root = int(cfg.WAV_OPTS.ROOT_IND)
        pg_db = 10.0 * np.log10(seq_len) if seq_len > 0 else float("nan")
        inter_zc_samps = seq_len + guard
        inter_zc_us = inter_zc_samps / fs * 1e6
        max_unambig_m = C_LIGHT * (inter_zc_samps / fs)
        lines += [
            "",
            f"ZC: SEQ_LEN={seq_len}  ROOT={root}  reps={n_rep}  guard={guard} samples",
            f"  Processing gain       : {pg_db:.2f} dB",
            f"  Inter-ZC spacing      : {inter_zc_us:.2f} us  ({inter_zc_samps} samples)",
            f"  Max unambig path delay: {inter_zc_us:.2f} us  -> {max_unambig_m:.1f} m one-way",
        ]

    n_fft = int(cfg.WAV_OPTS.N_FFT)
    cp_len = int(cfg.WAV_OPTS.CP_LEN)
    n_pilot = int(cfg.WAV_OPTS.N_PILOT)
    n_sub = int(cfg.WAV_OPTS.SUBCARRIERS)
    df_hz = fs / n_fft if n_fft > 0 else float("nan")
    sym_us = n_fft / fs * 1e6 if fs > 0 else float("nan")
    cp_us = cp_len / fs * 1e6 if fs > 0 else float("nan")
    cp_range_m = C_LIGHT * (cp_len / fs) if fs > 0 else float("nan")
    lines += [
        "",
        f"OFDM: N_FFT={n_fft}  CP={cp_len}  SUBCARRIERS={n_sub}  N_PILOTS={n_pilot}",
        f"  Subcarrier spacing    : {df_hz/1e3:.2f} kHz",
        f"  Useful symbol time    : {sym_us:.2f} us",
        f"  CP duration           : {cp_us:.2f} us",
        f"  Max delay spread (CP) : {cp_us:.2f} us  -> {cp_range_m:.1f} m one-way",
    ]

    return lines


def _expected_zc_peaks(cfg) -> list[int]:
    if cfg.WAVEFORM != "ZC":
        return []
    seq_len = int(cfg.WAV_OPTS.SEQ_LEN)
    n_rep = int(getattr(cfg.WAV_OPTS, "ZC_NUM_REPEATS", 4))
    guard = int(getattr(cfg.WAV_OPTS, "GUARD_LEN_SAMPS", 100))
    return [i * (seq_len + guard) for i in range(n_rep)]


def _xcorr_metrics(xcorr: np.ndarray, expected_peaks: list[int], ref_size: int):
    if xcorr.size == 0:
        return None
    peak = float(np.max(xcorr))
    if peak <= 0 or not expected_peaks:
        return {"peak": peak, "pslr_db": float("nan"), "sidelobe": float("nan")}
    mask = np.ones(xcorr.size, dtype=bool)
    half = max(1, ref_size // 2)
    for p in expected_peaks:
        lo = max(0, p - half)
        hi = min(xcorr.size, p + half)
        mask[lo:hi] = False
    sidelobe = float(np.max(xcorr[mask])) if np.any(mask) else 0.0
    pslr_db = 20.0 * np.log10(peak / sidelobe) if sidelobe > 0 else float("inf")
    return {"peak": peak, "pslr_db": pslr_db, "sidelobe": sidelobe}


def plot_waveform_preview(
    samples: np.ndarray,
    sections,
    sample_rate: float,
    ref_seq,
    expected_peaks: list[int],
    params_lines: list[str],
    out_path=None,
) -> None:
    n = samples.size
    duration_s = n / sample_rate if sample_rate > 0 else 0.0
    time_axis_us = np.arange(n) / sample_rate * 1e6 if sample_rate > 0 else np.arange(n).astype(float)
    magnitude = np.abs(samples)

    if n >= 256:
        spec_nperseg = 64
    elif n >= 128:
        spec_nperseg = 32
    else:
        spec_nperseg = max(16, n // 4)
    spec_noverlap = int(0.75 * spec_nperseg)

    spec_freqs, spec_times, spec = signal.stft(
        samples,
        fs=sample_rate,
        nperseg=spec_nperseg,
        noverlap=spec_noverlap,
        nfft=max(256, spec_nperseg),
        detrend=False,
        boundary=None,
        padded=False,
        return_onesided=False,
    )
    spec_freqs = np.fft.fftshift(spec_freqs)
    spec_mag = np.abs(np.fft.fftshift(spec, axes=0))
    spec_db = 20.0 * np.log10(np.maximum(spec_mag, 1e-12))
    spec_times_us = (spec_times - spec_times[0]) * 1e6 if spec_times.size > 0 else spec_times
    if spec_db.size > 0:
        vmin = float(np.percentile(spec_db, 5))
        vmax = float(np.percentile(spec_db, 99))
    else:
        vmin, vmax = -120.0, -20.0

    xcorr_t_us = None
    xcorr_db = None
    xcorr_metrics = None
    ref_size = 0
    if ref_seq is not None and ref_seq.size > 0 and samples.size >= ref_seq.size:
        ref_size = int(ref_seq.size)
        xcorr = np.abs(signal.correlate(samples, ref_seq, mode="valid", method="fft"))
        peak = float(np.max(xcorr)) if xcorr.size > 0 else 0.0
        floor = max(peak * 1e-6, 1e-12)
        xcorr_db = 20.0 * np.log10(np.maximum(xcorr, floor) / max(peak, 1e-12))
        xcorr_t_us = np.arange(xcorr.size) / sample_rate * 1e6
        xcorr_metrics = _xcorr_metrics(xcorr, expected_peaks, ref_size)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[3, 3, 3, 0.05],
        width_ratios=[3, 1.6],
        hspace=0.55,
        wspace=0.18,
    )
    fig.suptitle(
        f"Channel-Sounder Waveform Dry Run | N={n} | Fs={sample_rate/1e6:.3f} MHz | "
        f"T={duration_s*1e6:.2f} us",
        fontsize=12,
    )

    # Time-domain spans both columns of row 0
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(time_axis_us, np.real(samples), label="I", linewidth=0.8, alpha=0.7)
    ax_time.plot(time_axis_us, np.imag(samples), label="Q", linewidth=0.8, alpha=0.7)
    ax_time.plot(time_axis_us, magnitude, label="|x|", linewidth=1.4, color="black", alpha=0.8)
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlabel("Time (us)")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="upper right")

    edge = 0
    label_transform = mtransforms.blended_transform_factory(ax_time.transData, ax_time.transAxes)
    for name, length in sections:
        start_us = edge / sample_rate * 1e6
        edge += length
        end_us = edge / sample_rate * 1e6
        ax_time.axvline(start_us, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_time.text(
            (start_us + end_us) / 2.0,
            1.03,
            name,
            transform=label_transform,
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            clip_on=False,
        )
    ax_time.axvline(edge / sample_rate * 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Spectrogram 
    ax_spec = fig.add_subplot(gs[1, :])
    pcm = ax_spec.pcolormesh(
        spec_times_us,
        spec_freqs / 1e6,
        spec_db,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax_spec.set_ylabel("Frequency (MHz)")
    ax_spec.set_xlabel("Time (us)")
    ax_spec.set_title("Spectrogram")
    edge = 0
    for _name, length in sections:
        start_us = edge / sample_rate * 1e6
        edge += length
        ax_spec.axvline(start_us, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_spec.axvline(edge / sample_rate * 1e6, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.colorbar(pcm, ax=ax_spec, label="Magnitude (dB)")

    # Row 2 left: cross-correlation
    ax_xc = fig.add_subplot(gs[2, 0])
    if xcorr_db is not None:
        ax_xc.plot(xcorr_t_us, xcorr_db, linewidth=0.8)
        ax_xc.set_ylim(-60, 2)
        ax_xc.set_xlim(xcorr_t_us[0], xcorr_t_us[-1])
        for p in expected_peaks:
            t_p = p / sample_rate * 1e6
            if xcorr_t_us[0] <= t_p <= xcorr_t_us[-1]:
                ax_xc.axvline(t_p, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        title = "Cross-correlation against single ZC (peak-normalized)"
        if xcorr_metrics is not None and np.isfinite(xcorr_metrics.get("pslr_db", float("nan"))):
            title += f"  |  PSLR = {xcorr_metrics['pslr_db']:.1f} dB"
        ax_xc.set_title(title)
        ax_xc.set_xlabel("Lag in burst (us)")
        ax_xc.set_ylabel("|xcorr| (dB)")
        ax_xc.grid(True, alpha=0.25)
    else:
        ax_xc.text(0.5, 0.5, "No reference sequence available for this WAVEFORM",
                   ha="center", va="center", transform=ax_xc.transAxes, fontsize=10,
                   color="dimgray")
        ax_xc.set_axis_off()

    # Row 2 right: parameters 
    ax_params = fig.add_subplot(gs[2, 1])
    ax_params.axis("off")
    ax_params.text(
        0.0, 1.0, "\n".join(params_lines),
        family="monospace", fontsize=8.5,
        verticalalignment="top", horizontalalignment="left",
        transform=ax_params.transAxes,
    )
    ax_params.set_title("Configuration", loc="left", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=180)
        print(f"Saved waveform preview plot to: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run waveform generator and visualizer (no USRP required)."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="../config/tx_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="Save preview PNG here. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--save-npy",
        type=str,
        default="",
        help="Optional path to save generated complex waveform as .npy.",
    )

    args = parser.parse_args()

    if args.output:
        matplotlib.use("Agg")
    global plt
    import matplotlib.pyplot as _plt
    plt = _plt

    cfg = Config(args.config)
    wv = Waveform(cfg)
    waveform, sections = _build_waveform_sections(cfg, wv)
    ref = _detection_reference(cfg, wv)
    expected_peaks = _expected_zc_peaks(cfg)
    params_lines = _format_params(cfg, total_samples=int(waveform.size))

    out_path = None
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        _ensure_parent_dir(out_path)

    plot_waveform_preview(
        samples=waveform,
        sections=sections,
        sample_rate=float(cfg.USRP_CONF.SAMPLE_RATE),
        ref_seq=ref,
        expected_peaks=expected_peaks,
        params_lines=params_lines,
        out_path=out_path,
    )

    if args.save_npy:
        npy_path = Path(args.save_npy).expanduser().resolve()
        _ensure_parent_dir(npy_path)
        np.save(npy_path, waveform)
        print(f"Saved waveform samples to: {npy_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
