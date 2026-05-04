from dataclasses import dataclass
import pandas as pd


@dataclass
class CampaignResult:
    result_dir: str
    campaign_path: str
    meas: pd.DataFrame
    config: object
    freq: float
    wave_type: str


class CampaignCollection:
    _FIELD_MAP = {
        "resultDir": "result_dir",
        "meas": "meas",
        "config": "config",
        "freq": "freq",
        "waveType": "wave_type",
    }

    def __init__(self):
        self._campaigns: list[CampaignResult] = []

    def append(self, result: CampaignResult):
        self._campaigns.append(result)

    def __getitem__(self, key):
        if isinstance(key, str):
            attr = self._FIELD_MAP.get(key)
            if attr is None:
                raise KeyError(key)
            return [getattr(c, attr) for c in self._campaigns]
        return self._campaigns[key]

    def __len__(self):
        return len(self._campaigns)

    def __bool__(self):
        return len(self._campaigns) > 0

    def __iter__(self):
        return iter(self._campaigns)
