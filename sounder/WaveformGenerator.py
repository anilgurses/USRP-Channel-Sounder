import numpy as np

from scipy.signal import chirp, butter, sosfilt


class Waveform:
    # Make it more clear, add configs for waveforms
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _deterministic_qpsk(n, seed):
        rng = np.random.default_rng(int(seed))
        symbols = rng.integers(0, 4, size=int(n), dtype=np.int32)
        phase = (np.pi / 2.0) * symbols
        return np.exp(1j * phase).astype(np.complex64)

    def create_zadoff_chu(self):
        """
        Generates zadoff chu sequence

        Returns
        -----------
        zc_seq: np.complex64

        """
        import commpy

        zc_seq = commpy.zcsequence(
            self.config.WAV_OPTS.ROOT_IND, self.config.WAV_OPTS.SEQ_LEN
        )

        return zc_seq.astype(np.complex64)

    def create_OFDM(self):
        """
        Generate a single OFDM symbol with cyclic prefix.
        Note: This is experimental. I haven't fully tested it yet.

        Returns
        -----------
        OFDM_WAV: np.complex64
        """
        n_fft = int(self.config.WAV_OPTS.N_FFT)
        n_occupied = int(self.config.WAV_OPTS.SUBCARRIERS)
        n_pilot = int(self.config.WAV_OPTS.N_PILOT)
        cp_len = int(getattr(self.config.WAV_OPTS, "CP_LEN", n_fft // 4))
        dc_guard_bins = int(getattr(self.config.WAV_OPTS, "DC_GUARD_BINS", 10))
        edge_guard_bins = int(getattr(self.config.WAV_OPTS, "EDGE_GUARD_BINS", 0))

        if n_fft <= 0:
            raise ValueError("OFDM N_FFT must be positive.")
        if cp_len < 0 or cp_len >= n_fft:
            raise ValueError("OFDM CP_LEN must satisfy 0 <= CP_LEN < N_FFT.")
        if dc_guard_bins < 0 or edge_guard_bins < 0:
            raise ValueError("OFDM guard bins must be non-negative.")

        # Signed-carrier indexing around DC: [..., -2, -1, +1, +2, ...]
        # Keep Nyquist unallocated and reserve edge/DC guard bins.
        pos_start = dc_guard_bins + 1
        pos_stop = n_fft // 2 - edge_guard_bins
        positive = np.arange(pos_start, pos_stop, dtype=np.int32)
        if positive.size == 0:
            raise ValueError("No usable OFDM carriers remain after guard allocation.")

        available_occupied = positive.size * 2
        n_occupied = min(max(n_occupied, 0), available_occupied)
        if n_occupied % 2 != 0:
            n_occupied -= 1
        n_pilot = min(max(n_pilot, 0), n_occupied)
        if n_occupied == 0:
            raise ValueError("SUBCARRIERS resolves to zero active OFDM carriers.")

        n_neg = n_occupied // 2
        n_pos = n_occupied - n_neg

        occupied = np.concatenate(
            (-positive[:n_neg][::-1], positive[:n_pos]), axis=0
        )

        if n_pilot == occupied.size:
            pilot_offsets = occupied
        elif n_pilot == 0:
            pilot_offsets = np.array([], dtype=np.int32)
        else:
            pilot_idx = np.linspace(0, occupied.size - 1, n_pilot, dtype=np.int32)
            pilot_idx = np.unique(pilot_idx)
            if pilot_idx.size < n_pilot:
                missing = np.setdiff1d(
                    np.arange(occupied.size, dtype=np.int32),
                    pilot_idx,
                    assume_unique=True,
                )
                pilot_idx = np.concatenate(
                    (pilot_idx, missing[: n_pilot - pilot_idx.size]),
                    axis=0,
                )
            pilot_offsets = occupied[pilot_idx]

        ofdm_seed = int(getattr(self.config.WAV_OPTS, "SEED", 2026))
        normalize_peak = bool(getattr(self.config.WAV_OPTS, "NORMALIZE_PEAK", True))
        target_peak = float(getattr(self.config.WAV_OPTS, "TARGET_PEAK", 0.9))

        ofdm_bins = np.zeros(n_fft, dtype=np.complex64)
        if pilot_offsets.size > 0:
            pilot_values = self._deterministic_qpsk(pilot_offsets.size, ofdm_seed)
            ofdm_bins[pilot_offsets % n_fft] = pilot_values

        ofdm_symbol = np.fft.ifft(ofdm_bins, n=n_fft).astype(np.complex64)
        if normalize_peak:
            peak = float(np.max(np.abs(ofdm_symbol)))
            if peak > 0:
                ofdm_symbol = ofdm_symbol / peak
        ofdm_symbol = np.complex64(ofdm_symbol * target_peak)
        cyclic_prefix = ofdm_symbol[-cp_len:] if cp_len > 0 else np.empty(0, dtype=np.complex64)
        return np.concatenate((cyclic_prefix, ofdm_symbol), axis=0).astype(np.complex64)

    def create_GLFSR(self):
        """
        Generates PN sequence

        Returns
        -----------
        pn_seq : np.complex64

        """
        import galois

        poly = galois.Poly(self.config.WAV_OPTS.POLY)
        lfsr = galois.GLFSR(poly.reverse())
        pn_seq = lfsr.step(self.config.WAV_OPTS.SEQ_LEN)
        pn_seq = np.asarray(pn_seq)

        if self.config.WAV_OPTS.COMPLEX_BB:
            pn_seq = pn_seq + 1j * pn_seq

        return pn_seq

    def create_chirp(self):
        t_len = (
            int(self.config.WAV_OPTS.BW)
            if self.config.WAV_OPTS.COMPLEX
            else int(self.config.WAV_OPTS.BW * 2)
        )
        t = np.linspace(0, 1, t_len)
        t_1 = 1

        f_0 = 0
        if self.config.WAV_OPTS.COMPLEX:
            f_0 -= self.config.WAV_OPTS.BW / 2

        f_1 = 0 + self.config.WAV_OPTS.BW / 2

        beta = (f_1 - f_0) / t_1

        if self.config.WAV_OPTS.COMPLEX:
            i = np.cos(2 * np.pi * (t * t * beta / 2 + f_0 * t))
            q = np.cos(2 * np.pi * (t * t * beta / 2 + f_0 * t + 90 / 360))

            sig = np.complex64(i - q * 1j)

            return sig
        else:
            return np.cos(
                2
                * np.pi
                * (beta / 2 * t**2 + f_0 * t + self.config.WAV_OPTS.PHASE / 360)
            )

    def create_chirp_wav(self):
        w = self.create_chirp()

        if self.config.WAV_OPTS.COMPRESS:
            w_s = np.split(w, 100 / self.config.WAV_OPTS.PULSE_RATIO)
            w_s = np.array(w_s, dtype=np.complex64)
            w_s = np.sum(w_s, axis=0, dtype=np.complex64) / (
                100 / self.config.WAV_OPTS.PULSE_RATIO
            )

            return np.complex64(w_s)

        # if self.chirp_dur < 1:
        #     w[int(w.size*self.chirp_dur):] = 0
        # For other SDRs
        # w *= 2**14

        assert (
            self.config.WAV_OPTS.DURATION <= 1
        ), "Chirp can't be bigger than 1 second!"

        return np.complex64(w[: int(w.size * self.config.WAV_OPTS.DURATION)])

    def generate_preamble(self):
        # Cleanup
        barker_code = np.array(
            [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1] * 2, dtype=np.complex64
        )
        thresh = np.real(np.sum(barker_code * np.conjugate(barker_code)))

        int_frame = np.array([])
        for fr in barker_code:
            intr = np.zeros(self.sps, dtype=np.complex64)
            intr[0] = fr
            int_frame = np.concatenate((int_frame, intr))
        int_frame = np.complex64(int_frame)
        t = np.arange(-50, 51)
        h = (
            1
            / self.sps
            * np.sinc(t / self.sps)
            * np.cos(np.pi * self.roll_off * t / self.sps)
            / (1 - (2 * self.roll_off * t / self.sps) ** 2)
        )

        frame_filtd = np.convolve(int_frame, h)

        return frame_filtd, barker_code, thresh

    def create_waveform(self):
        wav_type = self.config.WAVEFORM

        wav = None
        guard = np.zeros(100, dtype=np.complex64)

        if wav_type == "PN":
            wav = self.create_GLFSR()
        elif wav_type == "ZC":
            wav = self.create_zadoff_chu()
            # wav = np.hstack([wav, wav, wav, wav])
            wav = np.concatenate((wav, wav, wav, wav),axis=0)
        elif wav_type == "CHIRP":
            wav = self.create_chirp_wav()
            
        ofdm = self.create_OFDM() 
        # wav = np.hstack([wav, guard, ofdm])
        wav = np.concatenate((wav, guard, ofdm),axis=0)


        # wav = np.tile(wav, int(1e3))

        if self.config.FILTER.ENABLED:
            sos = butter(
                12,
                int(self.config.FILTER.BW),
                "lp",
                fs=int(self.config.USRP_CONF.SAMPLE_RATE),
                output="sos",
            )
            wav = sosfilt(sos, wav)

        return wav

    def create_frame(self):
        # TODO Add feature for transmitting info
        # prb_intr, prb, thresh = self.generate_preamble()
        # frame = np.concatenate((prb_intr,crp))[:-prb_intr.size]

        return None
