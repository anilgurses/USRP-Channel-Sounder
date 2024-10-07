import numpy as np

from scipy.signal import chirp, butter, sosfilt
import commpy
import galois


class Waveform:
    # Make it more clear, add configs for waveforms
    def __init__(self, config):
        self.config = config

    def create_zadoff_chu(self):
        """
        Generates zadoff chu sequence

        Returns
        -----------
        zc_seq: np.complex64

        """
        zc_seq = commpy.zcsequence(
            self.config.WAV_OPTS.ROOT_IND, self.config.WAV_OPTS.SEQ_LEN
        )

        return zc_seq.astype(np.complex64)

    def create_OFDM(self):
        """
        Generated OFDM waveform with all of pilot symbols

        Returns
        -----------
        OFDM_WAV: np.complex64
        """
        K = self.config.WAV_OPTS.SUBCARRIERS  # Number of subcarriers
        P = self.config.WAV_OPTS.N_PILOT  # Number of pilot subcarriers
        N_FFT = self.config.WAV_OPTS.N_FFT
        CP = K // 4
        pilotValue = 1 + 1j
        pilotValue_2 = -1 + 0j
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

        pilotCarriers = allCarriers[:: K // P]  # Pilots is every (K/P)th carrier.
        s_pilotCarriers = pilotCarriers[::4]
        
        nullCarriers = [i-10 for i in range(20)]

        pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
        dataCarriers = np.delete(allCarriers, pilotCarriers)

        symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
        symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        symbol[s_pilotCarriers] = pilotValue_2  # allocate the pilot subcarriers
        symbol[dataCarriers] = 0 + 0j  # allocate the data subcarriers
        symbol[nullCarriers] = 0 + 0j  # allocate the null subcarriers

        OFDM_WAV = np.fft.ifft(symbol, N_FFT)

        cp = OFDM_WAV[-CP:]  # take the last CP samples ...
        OFDM_WAV = np.hstack([cp, OFDM_WAV])
        # Scale the signal
        # OFDM_WAV = OFDM_WAV / np.max(np.abs(OFDM_WAV))
        # print(np.max(OFDM_WAV))
        return OFDM_WAV.astype(np.complex64)

    def create_GLFSR(self):
        """
        Generates PN sequence

        Returns
        -----------
        pn_seq : np.complex64

        """
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
