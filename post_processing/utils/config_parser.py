import yaml
from dataclasses import dataclass

@dataclass
class USRP:
    SERIAL: str
    NUM_FRAMES: int = 32
    GAIN: int = 70
    SAMPLE_RATE: int = 20e6
    CENTER_FREQ: float = 3.4e9
    INIT_DELAY: float = 2.0
    CLK_REF: str = "INT"
    PPS_REF: str = "EXT"

@dataclass
class Calibration:
    TYPE: str = "INTERNAL"
    RX_REF: float = 0.0
    RX_REF_CSV: str = "../config/rx_pwr_ref.csv"
    TX_REF: float = 0.0
    TX_REF_CSV: str = "../config/rx_pwr_ref.csv"

@dataclass
class Interpolation:
    ENABLED: bool = False
    SPS: int = 2
    NUM_TAPS: int = 101
    ROLOFF: float = 0.35
    THR_OFF: int = 25

@dataclass
class Filter:
    ENABLED: bool = False
    TYPE: str = "LP"
    BW: float = 0.0


@dataclass
class GPS:
    ENABLED: bool = False
    SOURCE: str = ""
    DIR: str = ""


@dataclass
class RX_Opts:
    DURATION: float = 0.1
    POWER_CALC: bool = False
    PL_CALC: bool = False
    PLOT: bool = False
    OUTPUT_TYPE: str = "npz"


class Waveform:
    def __init__(self, type, wav_opts):
        self.type = type

        self.SUBCARRIERS = wav_opts["OFDM"]["SUBCARRIERS"]
        self.N_PILOT = wav_opts["OFDM"]["N_PILOTS"]
        self.N_FFT = wav_opts["OFDM"]["N_FFT"]

        if type == "PN":
            self.SEQ_LEN = wav_opts["PN"]["SEQ_LEN"]
            self.POLY = wav_opts["PN"]["POLY"]
            self.COMPLEX_BB = wav_opts["PN"]["COMPLEX_BB"]
        elif type == "ZC":
            self.SEQ_LEN = wav_opts["ZC"]["SEQ_LEN"]
            self.ROOT_IND = wav_opts["ZC"]["ROOT_IND"]
        elif type == "CHIRP":
            self.COMPLEX = wav_opts["CHIRP"]["COMPLEX"]
            self.PHASE = wav_opts["CHIRP"]["PHASE"]
            self.COMPRESS = wav_opts["CHIRP"]["COMPRESS"]
            self.PULSE_RATIO = wav_opts["CHIRP"]["PULSE_RATIO"]
            self.DURATION = wav_opts["CHIRP"]["DURATION"]
            self.BW = wav_opts["CHIRP"]["BW"]
        else:
            raise Exception("Waveform is not supported")


class Config(object):
    def __init__(self, fname):
        with open(fname, "r") as stream:
            config = yaml.safe_load(stream)
        self.raw = config 

        self.MODE = config["MODE"]
        self.PERIOD = config["PERIOD"]
        self.MAX_FREQ_OFF = config["MAX_FREQ_OFF"]

        temp_rx = config["RECV_OPTS"]
        self.RX = RX_Opts(
            temp_rx["DURATION"],
            temp_rx["CALC"]["POWER"],
            temp_rx["CALC"]["PL"],
            temp_rx["CALC"]["PLOT"],
            temp_rx["OUTPUT_TYPE"],
        )

        # Calibration settings
        # This is only for RX, since I have full uhd calibration
        # file for TX
        temp_cal = config["CALIBRATION"]
        self.CAL = Calibration(
            temp_cal["TYPE"],
            temp_cal["RX_REF"],
            temp_cal["RX_REF_CSV"],
            temp_cal["TX_REF"],
            temp_cal["TX_REF_CSV"],
        )

        temp_filter = config["FILTER"]
        self.FILTER = Filter(
            temp_filter["ENABLED"], temp_filter["TYPE"], temp_filter["BW"]
        )

        self.WAVEFORM = config["WAVEFORM"]
        # Waveform options
        self.WAV_OPTS = Waveform(self.WAVEFORM, config["WAV_OPTS"])

        # USRP related settings
        temp_usrp = config["USRP"]
        self.USRP_CONF = USRP(
            temp_usrp["SERIAL"],
            temp_usrp["NUM_FRAMES"],
            temp_usrp["GAIN"],
            temp_usrp["SAMPLE_RATE"],
            temp_usrp["CENTER_FREQ"],
            temp_usrp["INIT_DELAY"],
            temp_usrp["CLK_REF"],
            temp_usrp["PPS_REF"],
        )

        # Interpolation settings
        temp_interp = config["INTERPOLATION"]
        self.INTERP = Interpolation(
            temp_interp["ENABLED"],
            temp_interp["SPS"],
            temp_interp["NUM_TAPS"],
            temp_interp["ROLLOFF"],
            temp_interp["THR_OFF"],
        )

        temp_gps = config["GPS"]
        self.GPS = GPS(temp_gps["ENABLED"], temp_gps["SOURCE"], temp_gps["DIR"])
        self.NOTE = config["NOTE"]

    def to_dict(self):
        return self.raw
    