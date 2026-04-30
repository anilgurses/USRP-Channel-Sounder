import math


class PpsSlotScheduler:
    """Schedules burst slots relative to each integer PPS second."""

    def __init__(self, period_hz, first_epoch, tick_s=0.0002):
        self.period_hz = float(period_hz)
        self.slots_per_second = int(round(self.period_hz))
        if self.slots_per_second <= 0:
            raise ValueError("PERIOD must be positive")
        if abs(self.period_hz - self.slots_per_second) > 1e-9:
            raise ValueError("PPS slot scheduler requires integer PERIOD")

        self.slot_s = 1.0 / self.slots_per_second
        self.epoch_s = float(math.ceil(first_epoch))
        self.tick_s = float(tick_s)

    def quantize(self, value):
        if self.tick_s <= 0:
            return float(value)
        ticks = int(round(value / self.tick_s))
        return ticks * self.tick_s

    def time_for_index(self, slot_index):
        slot_index = int(slot_index)
        second_offset = slot_index // self.slots_per_second
        slot_offset = slot_index % self.slots_per_second
        return self.quantize(
            self.epoch_s + second_offset + slot_offset * self.slot_s
        )

    def next_index_after(self, usrp_time, min_index=0):
        rel = float(usrp_time) - self.epoch_s
        if rel < 0:
            candidate = 0
        else:
            candidate = int(math.floor(rel * self.slots_per_second)) + 1
        return max(int(min_index), candidate)
