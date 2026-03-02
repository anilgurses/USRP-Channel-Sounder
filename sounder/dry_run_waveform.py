import argparse
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt

from WaveformGenerator import Waveform
from utils.config_parser import Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_waveform_sections(cfg, wv: Waveform):
    if cfg.WAVEFORM == "PN":
        main = wv.create_GLFSR()
    elif cfg.WAVEFORM == "ZC":
        zc = wv.create_zadoff_chu()
        main = np.concatenate((zc, zc, zc, zc), axis=0)
    elif cfg.WAVEFORM == "CHIRP":
        main = wv.create_chirp_wav()
    else:
        raise ValueError(f"Unsupported waveform type: {cfg.WAVEFORM}")

    guard = np.zeros(100, dtype=np.complex64)
    ofdm = wv.create_OFDM()

    sections = [
        ("main", int(main.size)),
        ("guard", int(guard.size)),
        ("ofdm_tail", int(ofdm.size)),
    ]

    waveform = np.concatenate((main, guard, ofdm), axis=0).astype(np.complex64)

    if cfg.FILTER.ENABLED:
        sos = butter(
            12,
            int(cfg.FILTER.BW),
            "lp",
            fs=int(cfg.USRP_CONF.SAMPLE_RATE),
            output="sos",
        )
        waveform = sosfilt(sos, waveform).astype(np.complex64)

    return waveform, sections


def plot_waveform_preview(samples: np.ndarray, sections, sample_rate: float, out_path: Path) -> None:
    n = samples.size
    duration_s = n / sample_rate if sample_rate > 0 else 0.0
    time_axis_s = np.arange(n) / sample_rate if sample_rate > 0 else np.arange(n)
    time_axis_us = time_axis_s * 1e6
    magnitude = np.abs(samples)

    # STFT tuned for short timing features so guard/tail are visible.
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

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Waveform Dry Run | N={n} | Fs={sample_rate:.3f} Hz | T={duration_s*1e6:.3f} us",
        fontsize=12,
    )

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(time_axis_us, np.real(samples), label="I", linewidth=0.8, alpha=0.7)
    ax1.plot(time_axis_us, np.imag(samples), label="Q", linewidth=0.8, alpha=0.7)
    ax1.plot(time_axis_us, magnitude, label="|x|", linewidth=1.4, color="black", alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (us)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    # Draw section boundaries/labels.
    edge = 0
    label_transform = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    for name, length in sections:
        start_us = edge / sample_rate * 1e6
        edge += length
        end_us = edge / sample_rate * 1e6
        ax1.axvline(start_us, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
        ax1.text(
            (start_us + end_us) / 2.0,
            1.03,
            name,
            transform=label_transform,
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            clip_on=False,
        )
    ax1.axvline(edge / sample_rate * 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)

    ax2 = fig.add_subplot(2, 1, 2)
    pcm = ax2.pcolormesh(
        spec_times_us,
        spec_freqs / 1e6,
        spec_db,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_ylabel("Frequency (MHz)")
    ax2.set_xlabel("Time (us)")
    ax2.set_title("Spectrogram")

    edge = 0
    for _name, length in sections:
        start_us = edge / sample_rate * 1e6
        edge += length
        ax2.axvline(start_us, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axvline(edge / sample_rate * 1e6, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.colorbar(pcm, ax=ax2, label="Magnitude (dB)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run waveform generator and visualizer (no USRP required)."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../config/tx_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../plots/waveform_dry_run.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--save-npy",
        type=str,
        default="",
        help="Optional path to save generated complex waveform as .npy.",
    )

    args = parser.parse_args()

    cfg = Config(args.config)
    waveform, sections = _build_waveform_sections(cfg, Waveform(cfg))

    out_path = Path(args.output).expanduser().resolve()
    _ensure_parent_dir(out_path)
    plot_waveform_preview(
        samples=waveform,
        sections=sections,
        sample_rate=float(cfg.USRP_CONF.SAMPLE_RATE),
        out_path=out_path,
    )

    if args.save_npy:
        npy_path = Path(args.save_npy).expanduser().resolve()
        _ensure_parent_dir(npy_path)
        np.save(npy_path, waveform)
        print(f"Saved waveform samples to: {npy_path}")

    print(f"Saved waveform preview plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
