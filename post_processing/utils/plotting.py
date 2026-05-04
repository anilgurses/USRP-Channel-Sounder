import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale


class PostProcessorPlots:
    def _require_data(self):
        if not self.pp_data:
            print("No processed data found. Please run process_date/s() first.")
            return False
        return True

    def _get_campaign_df(self, index):
        return self.pp_data["meas"][index]

    @staticmethod
    def _campaign_id_from_result_dir(result_dir):
        return os.path.basename(result_dir.rstrip("/"))

    def _campaign_label(self, index):
        df = self.pp_data["meas"][index]
        freq = self.pp_data["freq"][index]
        alt = float(np.nanpercentile(df["alt"], 90)) if len(df) else np.nan
        cid = self._campaign_id_from_result_dir(self.pp_data["resultDir"][index])
        if np.isfinite(alt):
            return f"{cid} ({int(round(freq))} MHz, {int(round(alt))} m)"
        return cid

    def get_colorscale(self):
        x = np.linspace(0, 1, 50)

        def list_rgba_colors(alpha=0.1):
            colors = sample_colorscale("jet", list(x))
            return [c.replace("rgb", "rgba").replace(")", f", {alpha})") for c in colors]

        return list_rgba_colors(0.15), list_rgba_colors(1.0)

    # --- generic multi-campaign comparison helpers ---

    def _plot_multiple_metric_aligned_by_location(self, column, ylabel, title,
                                                  category, filename, save=False):
        if not self._require_data():
            return
        if len(self.pp_data["meas"]) < 2:
            print("Need at least two campaigns for comparison.")
            return

        ref_index = None
        ref_len = -np.inf
        ref_data = None
        for i, df in enumerate(self.pp_data["meas"]):
            if len(df) < 2:
                continue
            candidate = self._build_reference_route(df)
            route_len = candidate[3][-1] if len(candidate[3]) else -np.inf
            if route_len > ref_len:
                ref_index = i
                ref_len = route_len
                ref_data = candidate
        if ref_index is None or ref_data is None:
            print("Could not build a reference route for comparison.")
            return

        _, x_ref, y_ref, ref_pos, lat0, lon0 = ref_data
        bin_width_m = max(ref_len / 150.0, 2.0)

        plt.figure()
        for i in range(len(self.pp_data["meas"])):
            aligned = self._align_to_reference_route(
                self.pp_data["meas"][i], x_ref, y_ref, ref_pos, lat0, lon0
            )
            x_prof, y_prof = self._binned_track_profile(
                aligned["aligned_pos_m"], aligned[column], bin_width_m
            )
            if len(x_prof) == 0:
                continue
            plt.plot(x_prof, y_prof, lw=1.8, label=self._campaign_label(i))

        ref_cid = self._campaign_id_from_result_dir(self.pp_data["resultDir"][ref_index])
        plt.xlabel("Aligned Route Position (m)")
        plt.ylabel(ylabel)
        plt.title(f"{title} — Location-Aligned to {ref_cid}")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(self._comparison_fig_path(category, filename),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_multiple_metric_vs_time(self, column, ylabel, title,
                                      category, filename, save=False):
        if not self._require_data():
            return
        if len(self.pp_data["meas"]) < 2:
            print("Need at least two campaigns for comparison.")
            return

        plt.figure()
        for i, df in enumerate(self.pp_data["meas"]):
            valid = df["time"].notna() & df[column].notna()
            if not valid.any():
                continue
            plt.plot(df.loc[valid, "time"], df.loc[valid, column],
                     lw=1.2, alpha=0.85, label=self._campaign_label(i))

        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(self._comparison_fig_path(category, filename),
                        dpi=150, bbox_inches="tight")
        plt.show()

    # --- per-campaign plots ---

    def plot_time_vs_power(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.plot(df["time"], df["avgPower"])
        plt.xlabel("Time (s)")
        plt.ylabel("Received Power (dBFS)")
        plt.title("Received Power vs Time")
        if save:
            plt.savefig(self._fig_path(index, "power", "power_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_multiple_time_vs_power(self, save=False):
        self._plot_multiple_metric_vs_time(
            "avgPower",
            "Received Power (dBFS)",
            "Received Power-All Campaigns",
            "power",
            "multiple_power_vs_time.png",
            save=save,
        )

    def plot_multiple_power_vs_location(self, save=False):
        self._plot_multiple_metric_aligned_by_location(
            "avgPower",
            "Received Power (dBFS)",
            "Received Power-All Campaigns",
            "power",
            "multiple_power_vs_location_aligned.png",
            save=save,
        )

    def plot_dist_vs_power(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.scatter(df["dist"], df["avgPower"], s=4, alpha=0.5)
        plt.xlabel("Distance (m)")
        plt.ylabel("Received Power (dBFS)")
        plt.title("Received Power vs Distance")
        if save:
            plt.savefig(self._fig_path(index, "power", "power_vs_dist.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_time_vs_loc(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="avgPower",
                                zoom=15, labels={"avgPower": "Power (dBFS)"})
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()
        if save:
            fig.write_html(self._fig_path(index, "power", "power_vs_loc.html"))

    def plot_multiple_time_vs_loc(self, save=False):
        if not self._require_data():
            return
        fig = go.Figure()
        for i in range(len(self.pp_data["meas"])):
            df = self.pp_data["meas"][i]
            fig.add_trace(go.Scattermapbox(
                lat=df["lat"], lon=df["lon"], mode="markers",
                marker=go.scattermapbox.Marker(
                    size=14, color=df["avgPower"],
                    colorscale="Jet", opacity=0.7,
                ),
                text=df["time"], name=f"Experiment {i}",
            ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=go.layout.mapbox.Center(lat=35.773851, lon=-78.677010), zoom=10),
        )
        fig.show()
        if save:
            fig.write_html(self._comparison_fig_path("power", "multiple_power_vs_loc.html"))

    def plot_freqoff_vs_time(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.plot(df["time"], df["freq_offset"])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency Offset (Hz)")
        plt.title("Frequency Offset vs Time")
        if save:
            plt.savefig(self._fig_path(index, "frequency", "freq_offset_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_multiple_freqoff_vs_time(self, save=False):
        self._plot_multiple_metric_vs_time(
            "freq_offset",
            "Frequency Offset (Hz)",
            "Frequency Offset — All Campaigns",
            "frequency",
            "multiple_freq_offset_vs_time.png",
            save=save,
        )

    def plot_multiple_freqoff_vs_location(self, save=False):
        self._plot_multiple_metric_aligned_by_location(
            "freq_offset",
            "Frequency Offset (Hz)",
            "Frequency Offset — All Campaigns",
            "frequency",
            "multiple_freq_offset_vs_location_aligned.png",
            save=save,
        )

    def plot_snr_vs_time(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.plot(df["time"], df["avgSnr"])
        plt.xlabel("Time (s)")
        plt.ylabel("SNR (dB)")
        plt.title("SNR vs Time")
        if save:
            plt.savefig(self._fig_path(index, "snr", "snr_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_multiple_snr_vs_time(self, save=False):
        self._plot_multiple_metric_vs_time(
            "avgSnr",
            "SNR (dB)",
            "SNR — All Campaigns",
            "snr",
            "multiple_snr_vs_time.png",
            save=save,
        )

    def plot_multiple_snr_vs_location(self, save=False):
        self._plot_multiple_metric_aligned_by_location(
            "avgSnr",
            "SNR (dB)",
            "SNR — All Campaigns",
            "snr",
            "multiple_snr_vs_location_aligned.png",
            save=save,
        )

    def plot_multiple_pathloss_vs_location(self, save=False):
        self._plot_multiple_metric_aligned_by_location(
            "avg_pl",
            "Path Loss (dB)",
            "Path Loss — All Campaigns",
            "path_loss",
            "multiple_path_loss_vs_location_aligned.png",
            save=save,
        )

    def plot_dist_vs_snr(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.scatter(df["dist"], df["avgSnr"], s=4, alpha=0.5)
        plt.xlabel("Distance (m)")
        plt.ylabel("SNR (dB)")
        plt.title("SNR vs Distance")
        if save:
            plt.savefig(self._fig_path(index, "snr", "snr_vs_dist.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_snr_vs_loc(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="avgSnr", zoom=15,
                                color_continuous_scale="Jet",
                                labels={"avgSnr": "SNR (dB)"})
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()
        if save:
            fig.write_html(self._fig_path(index, "snr", "snr_vs_loc.html"))

    def plot_doppler_spectrum(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        if "doppler_spectrum" not in df.columns:
            return
        valid = [s for s in df["doppler_spectrum"] if hasattr(s, "__len__") and len(s) > 1]
        if not valid:
            return
        avg_psd     = np.mean(valid, axis=0)
        sample_rate = self.pp_data["config"][index].USRP_CONF.SAMPLE_RATE
        nperseg     = 1024
        freq_axis   = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / sample_rate))
        plt.figure()
        plt.plot(freq_axis, 10 * np.log10(np.fft.fftshift(avg_psd)))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB)")
        plt.title(f"Avg Wideband PSD  (mean Doppler shift: {df['doppler_shift'].mean():.1f} Hz)")
        if save:
            plt.savefig(self._fig_path(index, "doppler", "doppler_spectrum.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_multiple_doppler_spectrum(self, save=False):
        if not self._require_data():
            return
        plt.figure()
        for i in range(len(self.pp_data["meas"])):
            df = self.pp_data["meas"][i]
            if "doppler_spectrum" not in df.columns:
                continue
            valid = [s for s in df["doppler_spectrum"] if hasattr(s, "__len__") and len(s) > 1]
            if not valid:
                continue
            avg_psd     = np.mean(valid, axis=0)
            sample_rate = self.pp_data["config"][i].USRP_CONF.SAMPLE_RATE
            nperseg     = 1024
            freq_axis   = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / sample_rate))
            plt.plot(freq_axis, 10 * np.log10(np.fft.fftshift(avg_psd)),
                     label=f"Exp {i} (shift: {df['doppler_shift'].mean():.1f} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB)")
        plt.title("Avg Wideband PSD — All Campaigns")
        plt.legend()
        if save:
            plt.savefig(self._comparison_fig_path("doppler", "multiple_doppler_spectrum.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_rms_delay_spread_vs_dist(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        rds = df["rms_delay_spread"] * 1e9
        ok  = rds > 0
        plt.figure()
        plt.scatter(df["dist"][ok], np.log10(rds[ok]), s=4, alpha=0.5)
        plt.xlabel("Distance (m)")
        plt.ylabel("log\u2081\u2080(RMS Delay Spread (ns))")
        plt.title("RMS Delay Spread vs Distance")
        if save:
            plt.savefig(self._fig_path(index, "delay_spread", "rms_ds_vs_dist.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_rms_delay_spread_vs_time(self, index, save=False, offset=0):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        rds = (df["rms_delay_spread"] + offset) * 1e9
        ok  = rds > 0
        plt.figure()
        plt.plot(df["time"][ok], np.log10(rds[ok]))
        plt.xlabel("Time (s)")
        plt.ylabel("log\u2081\u2080(RMS Delay Spread (ns))")
        plt.title("RMS Delay Spread vs Time")
        if save:
            plt.savefig(self._fig_path(index, "delay_spread", "rms_ds_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_rms_delay_spread_cdf(self, index, save=False, offset=0):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        rds = df["rms_delay_spread"].dropna()
        rds = rds[np.isfinite(rds)]
        sorted_ns = (np.sort(rds) + offset) * 1e9
        sorted_ns = sorted_ns[sorted_ns > 0]
        if len(sorted_ns) == 0:
            print("No positive RMS delay spread values to plot.")
            return
        yvals = np.arange(1, len(sorted_ns) + 1) / len(sorted_ns)
        plt.figure()
        plt.plot(np.log10(sorted_ns), yvals)
        plt.xlabel("log\u2081\u2080(RMS Delay Spread (ns))")
        plt.ylabel("CDF")
        plt.title("RMS Delay Spread CDF")
        plt.ylim([0, 1])
        if save:
            plt.savefig(self._fig_path(index, "delay_spread", "rms_ds_cdf.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_rms_delay_spread_cdf_vs_altitude(self, index, num_groups, save=False,
                                               offset=0, use_log10_x_axis=True):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        bounds = np.linspace(df["alt"].min(), df["alt"].max(), num_groups + 1)
        plt.figure(figsize=(10, 6))
        for i in range(num_groups):
            lo, hi = bounds[i], bounds[i + 1]
            mask   = (df["alt"] >= lo) & (df["alt"] <= hi if i == num_groups - 1 else df["alt"] < hi)
            rds    = df.loc[mask, "rms_delay_spread"].dropna()
            rds    = rds[np.isfinite(rds)]
            sorted_ns = (np.sort(rds) + offset) * 1e9
            sorted_ns = sorted_ns[sorted_ns > 0]
            if len(sorted_ns) == 0:
                continue
            cdf    = np.arange(1, len(sorted_ns) + 1) / len(sorted_ns)
            xvals  = np.log10(sorted_ns) if use_log10_x_axis else sorted_ns
            plt.plot(xvals, cdf, label=f"{lo:.0f}\u2013{hi:.0f} m")
        plt.xlabel("log10(RMS Delay Spread (ns))" if use_log10_x_axis else "RMS Delay Spread (ns)")
        plt.ylabel("CDF")
        plt.title("RMS Delay Spread CDF by Altitude")
        plt.legend()
        plt.ylim([0, 1])
        plt.gca().set_facecolor("none")
        plt.gcf().patch.set_alpha(0)
        if save:
            fname = ("rms_ds_cdf_vs_altitude.png" if use_log10_x_axis
                     else "rms_ds_cdf_vs_altitude_linear_x.png")
            plt.savefig(self._fig_path(index, "delay_spread", fname),
                        dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_k_factor_vs_dist(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.scatter(df["dist"], df["k_factor"], s=4, alpha=0.5)
        plt.xlabel("Distance (m)")
        plt.ylabel("K-Factor (dB)")
        plt.title("Rician K-Factor vs Distance")
        if save:
            plt.savefig(self._fig_path(index, "k_factor", "k_factor_vs_dist.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_k_factor_vs_time(self, index, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        plt.figure()
        plt.plot(df["time"], df["k_factor"])
        plt.xlabel("Time (s)")
        plt.ylabel("K-Factor (dB)")
        plt.title("Rician K-Factor vs Time")
        if save:
            plt.savefig(self._fig_path(index, "k_factor", "k_factor_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def calculate_path_loss_exponent(self, index, plot=False, save=False):
        if not self._require_data():
            return None, None
        df = self._get_campaign_df(index)
        df_f = df[df["avg_pl"].notna() & (df["dist"] > 1)]
        if df_f.empty:
            print("No valid data to calculate path loss exponent.")
            return None, None
        n, pl0 = self._fit_path_loss_model(df_f["dist"], df_f["avg_pl"])
        print(f"Path loss exponent n = {n:.3f},  PL(1m) = {pl0:.2f} dB")
        if plot:
            self.plot_path_loss_fit(index, n, pl0, save=save)
        return n, pl0

    def plot_path_loss_fit(self, index, path_loss_exponent, intercept, save=False):
        df   = self._get_campaign_df(index)
        df_f = df[df["avg_pl"].notna() & (df["dist"] > 1)]
        dist = df_f["dist"].to_numpy(dtype=float)
        x_fit = np.geomspace(dist.min(), dist.max(), 200)
        plt.figure()
        plt.scatter(dist, df_f["avg_pl"], s=6, alpha=0.4, label="Measured")
        plt.plot(x_fit, intercept + 10.0 * path_loss_exponent * np.log10(x_fit),
                 "r-", lw=2,
                 label=f"Fit: n={path_loss_exponent:.2f}, PL(1m)={intercept:.1f} dB")
        plt.xscale("log")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        plt.xlabel("Distance (m)")
        plt.ylabel("Path Loss (dB)")
        plt.title("Path Loss vs Distance")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(self._fig_path(index, "path_loss", "pl_fit.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_path_loss_exponent_vs_altitude(self, index, num_groups, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        bounds = np.linspace(df["alt"].min(), df["alt"].max(), num_groups + 1)
        ple_vals, alt_centers = [], []
        for i in range(num_groups):
            lo, hi = bounds[i], bounds[i + 1]
            mask   = (df["alt"] >= lo) & (df["alt"] <= hi if i == num_groups - 1 else df["alt"] < hi)
            df_g   = df[mask & df["avg_pl"].notna() & (df["dist"] > 1)]
            if len(df_g) < 2:
                continue
            n, _ = self._fit_path_loss_model(df_g["dist"], df_g["avg_pl"])
            ple_vals.append(n)
            alt_centers.append((lo + hi) / 2)
        if not ple_vals:
            print("Could not calculate path loss exponent for any altitude group.")
            return
        plt.figure()
        plt.plot(alt_centers, ple_vals, marker="o")
        plt.xlabel("Altitude (m)")
        plt.ylabel("Path Loss Exponent (n)")
        plt.title("Path Loss Exponent vs Altitude")
        plt.grid(True)
        if save:
            plt.savefig(self._fig_path(index, "path_loss", "pl_exponent_vs_altitude.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    # --- new plots: distance comparison ---

    def plot_est_dist_vs_gps_dist(self, index, save=False):
        """Scatter of cross-correlation estimated distance vs GPS geodesic distance."""
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        ok = df["est_dist"].notna() & df["dist"].notna() & (df["dist"] > 0)
        d = df[ok]
        if d.empty:
            print("No valid distance data to plot.")
            return

        fig, ax = plt.subplots()
        ax.scatter(d["dist"], d["est_dist"], s=4, alpha=0.5, label="Measurements")

        lims = [min(d["dist"].min(), d["est_dist"].min()),
                max(d["dist"].max(), d["est_dist"].max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Ideal (1:1)")

        ax.set_xlabel("GPS Distance (m)")
        ax.set_ylabel("Estimated Distance (m)")
        ax.set_title("Estimated Distance vs GPS Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(self._fig_path(index, "distance", "est_dist_vs_gps_dist.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_dist_comparison_vs_time(self, index, save=False):
        """Plot GPS distance and estimated distance on the same time axis."""
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        ok = df["time"].notna() & df["dist"].notna() & df["est_dist"].notna()
        d = df[ok]
        if d.empty:
            print("No valid distance data to plot.")
            return

        fig, ax = plt.subplots()
        ax.plot(d["time"], d["dist"], lw=1.2, label="GPS Distance")
        ax.plot(d["time"], d["est_dist"], lw=1.2, alpha=0.8, label="Estimated Distance")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("GPS vs Estimated Distance over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(self._fig_path(index, "distance", "dist_comparison_vs_time.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    def plot_cir(self, index, meas_index, sigmf=None, normalize=True,
                 db=True, save=False):
        metric = self.reprocess_single_cir(index, meas_index, sigmf=sigmf)
        if metric is None or len(metric.corr) == 0:
            print("Could not retrieve CIR for this measurement.")
            return

        cir = metric.corr
        if normalize:
            cir = cir / np.max(np.abs(cir))

        sample_rate = self.pp_data["config"][index].USRP_CONF.SAMPLE_RATE
        time_axis = np.arange(len(cir)) / sample_rate * 1e6  # microseconds

        if db:
            magnitude = 20 * np.log10(np.abs(cir) + 1e-12)
            ylabel = "Magnitude (dB)"
        else:
            magnitude = np.abs(cir)
            ylabel = "Magnitude"

        fig, ax = plt.subplots()
        ax.plot(time_axis, magnitude, lw=0.8)
        ax.set_xlabel("Delay (\u00b5s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"CIR @ d={metric.dist:.0f} m, t={metric.time:.2f} s")
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(self._fig_path(index, "cir", f"cir_meas_{meas_index}.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()
        return metric

    def plot_cir_waterfall(self, index, n_samples=10, sigmf=None,
                           normalize=True, save=False):
        if not self._require_data():
            return
        df = self._get_campaign_df(index)
        total = len(df)
        if total == 0:
            print("No measurements to plot.")
            return
        indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)

        sample_rate = self.pp_data["config"][index].USRP_CONF.SAMPLE_RATE

        fig, axes = plt.subplots(len(indices), 1, figsize=(12, 2.5 * len(indices)),
                                 sharex=True)
        if len(indices) == 1:
            axes = [axes]

        for ax_idx, mi in enumerate(indices):
            metric = self.reprocess_single_cir(index, int(mi), sigmf=sigmf)
            if metric is None or len(metric.corr) == 0:
                axes[ax_idx].text(0.5, 0.5, f"Meas {mi}: no CIR",
                                  transform=axes[ax_idx].transAxes, ha="center")
                continue

            cir = metric.corr
            if normalize:
                cir = cir / np.max(np.abs(cir))
            magnitude = 20 * np.log10(np.abs(cir) + 1e-12)
            time_axis = np.arange(len(cir)) / sample_rate * 1e6

            axes[ax_idx].plot(time_axis, magnitude, lw=0.6)
            axes[ax_idx].set_ylabel("dB")
            label = f"d={metric.dist:.0f} m" if hasattr(metric, "dist") else f"#{mi}"
            axes[ax_idx].set_title(label, fontsize=9, loc="left")
            axes[ax_idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Delay (\u00b5s)")
        fig.suptitle("CIR Waterfall", fontsize=12)
        fig.tight_layout()
        if save:
            fig.savefig(self._fig_path(index, "cir", "cir_waterfall.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()

    # ---orchestrator ---

    def generate_all_figures(self, altitude_groups=4):
        """Generate and save every figure for all loaded campaigns."""
        if not self._require_data():
            return

        was_interactive = plt.isinteractive()
        plt.ioff()
        n = len(self.pp_data["meas"])

        for i in range(n):
            cid = os.path.basename(self.pp_data["resultDir"][i].rstrip("/"))
            print(f"[{i + 1}/{n}] Figures for {cid} ...")

            self.plot_time_vs_power(i, save=True)
            self.plot_dist_vs_power(i, save=True)
            self.plot_time_vs_loc(i, save=True)
            self.plot_freqoff_vs_time(i, save=True)
            self.plot_snr_vs_time(i, save=True)
            self.plot_dist_vs_snr(i, save=True)
            self.plot_snr_vs_loc(i, save=True)
            self.plot_doppler_spectrum(i, save=True)
            self.plot_rms_delay_spread_vs_dist(i, save=True)
            self.plot_rms_delay_spread_vs_time(i, save=True)
            self.plot_rms_delay_spread_cdf(i, save=True)
            self.plot_rms_delay_spread_cdf_vs_altitude(i, altitude_groups, save=True)
            self.plot_k_factor_vs_dist(i, save=True)
            self.plot_k_factor_vs_time(i, save=True)
            self.calculate_path_loss_exponent(i, plot=True, save=True)
            self.plot_path_loss_exponent_vs_altitude(i, altitude_groups, save=True)
            self.plot_est_dist_vs_gps_dist(i, save=True)
            self.plot_dist_comparison_vs_time(i, save=True)

            plt.close("all")
            print(f"    -> {self.pp_data['resultDir'][i]}figures/")

        if n > 1:
            print("Generating comparison figures ...")
            self.plot_multiple_time_vs_power(save=True)
            self.plot_multiple_power_vs_location(save=True)
            self.plot_multiple_freqoff_vs_time(save=True)
            self.plot_multiple_freqoff_vs_location(save=True)
            self.plot_multiple_snr_vs_time(save=True)
            self.plot_multiple_snr_vs_location(save=True)
            self.plot_multiple_pathloss_vs_location(save=True)
            self.plot_multiple_doppler_spectrum(save=True)
            self.plot_multiple_time_vs_loc(save=True)
            plt.close("all")
            print(f"    -> {os.path.join(os.path.dirname(self.pp_data['resultDir'][0]), '..', 'comparison')}/")

        if was_interactive:
            plt.ion()
        print("All figures saved.")
