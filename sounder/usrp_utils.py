import math
import shutil
import subprocess
import time

import uhd


def align_device_time_to_utc(usrp, logger, terminate=None,
                              max_retries=3, edge_slack_s=0.3):
    for attempt in range(1, max_retries + 1):
        _raise_if_terminating(terminate)
        now = time.time()
        next_pps_at = math.ceil(now)
        slack = next_pps_at - now
        if slack < edge_slack_s:
            # Too close to the next edge
            time.sleep(slack + 0.05)
            continue
        target_utc = next_pps_at  # device clock value AT the next PPS edge
        usrp.set_time_next_pps(uhd.types.TimeSpec(float(target_utc)))
        wait_s = max(0.0, target_utc - time.time()) + 0.2
        time.sleep(wait_s)
        observed = usrp.get_time_last_pps().get_real_secs()
        if abs(observed - target_utc) < 0.5:
            logger.info(
                "device time UTC-aligned: target=%d, last_pps=%.3f, host=%.3f",
                target_utc, observed, time.time(),
            )
            return
        logger.warn(
            "UTC alignment attempt %d: target=%d last_pps=%.3f (off by %.3fs); retrying",
            attempt, target_utc, observed, observed - target_utc,
        )
    raise RuntimeError(
        f"UTC alignment failed after {max_retries} attempts (last observed="
        f"{observed:.3f} vs target={target_utc}); check host NTP / GPSDO PPS wiring"
    )


def check_host_clock_sync(logger):
    """
    The UTC alignment relies on `time.time()` being accurate to better than
    0.5 s. If chrony/timedatectl says otherwise, log loudly  
    """
    if shutil.which("timedatectl"):
        try:
            out = subprocess.run(
                ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                check=False, capture_output=True, text=True, timeout=2,
            )
            value = (out.stdout or "").strip().lower()
            if value == "yes":
                logger.info("host clock: NTPSynchronized=yes")
                return
            logger.warn("host clock NOT NTP-synchronised (timedatectl: %r). "
                        "UTC alignment may land on the wrong second.", value or "no output")
            return
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warn("could not query timedatectl: %s", exc)
    if shutil.which("chronyc"):
        try:
            out = subprocess.run(
                ["chronyc", "tracking"], check=False, capture_output=True,
                text=True, timeout=2,
            )
            logger.info("chronyc tracking:\n%s", (out.stdout or out.stderr).strip())
            return
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warn("could not query chronyc: %s", exc)
    logger.warn("no NTP query tool found (timedatectl / chronyc); cannot verify host clock")


def configure_role(usrp, config, role, rx_subdev=None, tx_subdev=None):
    role = role.upper()
    if role == "RX":
        subdev = rx_subdev or config.USRP_CONF.RX_SUBDEV
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(subdev))
        rx_gain = (config.USRP_CONF.RX_GAIN
                   if config.USRP_CONF.RX_GAIN is not None
                   else config.USRP_CONF.GAIN)
        usrp.set_rx_rate(config.USRP_CONF.SAMPLE_RATE, 0)
        usrp.clear_command_time()
        usrp.set_command_time(usrp.get_time_now() + uhd.types.TimeSpec(0.1))
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)
        usrp.set_rx_gain(float(rx_gain), 0)
        usrp.clear_command_time()
    elif role == "TX":
        subdev = tx_subdev or config.USRP_CONF.TX_SUBDEV
        usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(subdev))
        tx_gain = (config.USRP_CONF.TX_GAIN
                   if config.USRP_CONF.TX_GAIN is not None
                   else config.USRP_CONF.GAIN)
        usrp.set_tx_rate(config.USRP_CONF.SAMPLE_RATE, 0)
        usrp.clear_command_time()
        usrp.set_command_time(usrp.get_time_now() + uhd.types.TimeSpec(0.1))
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(config.USRP_CONF.CENTER_FREQ), 0)
        usrp.set_tx_gain(float(tx_gain), 0)
        usrp.clear_command_time()
    else:
        raise ValueError(f"unknown role {role!r}; expected 'TX' or 'RX'")
    time.sleep(0.1)  # LO settle


def createMultiUSRP(config):
    dev_str = f"num_recv_frames={config.USRP_CONF.NUM_FRAMES},num_send_frames={config.USRP_CONF.NUM_FRAMES}"
    dev_str += f",serial={config.USRP_CONF.SERIAL}" if config.USRP_CONF.SERIAL else ""
    usrp = uhd.usrp.MultiUSRP(dev_str)
    return usrp


def _raise_if_terminating(terminate=None):
    if terminate is not None and terminate.is_set():
        raise KeyboardInterrupt


def _wait_for_sensor_true(
    usrp,
    sensor_name,
    logger,
    board=0,
    timeout_s=120,
    sleep_s=0.2,
    terminate=None,
):
    deadline = time.time() + timeout_s
    while not usrp.get_mboard_sensor(sensor_name, board).to_bool():
        _raise_if_terminating(terminate)
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for sensor '{sensor_name}' lock.")
        logger.info("Waiting for %s lock", sensor_name)
        time.sleep(sleep_s)


def _wait_for_first_pps(usrp, logger, terminate=None):
    last_pps_time = usrp.get_time_last_pps().get_real_secs()
    while last_pps_time == usrp.get_time_last_pps().get_real_secs():
        _raise_if_terminating(terminate)
        logger.info("Waiting for first PPS")
        time.sleep(0.1)


def init_clock_pps_sources(config, usrp, logger, terminate=None):
    check_host_clock_sync(logger)
    if config.USRP_CONF.CLK_REF == "GPSDO":
        usrp.set_clock_source("gpsdo")
        _wait_for_sensor_true(usrp, "ref_locked", logger,
                              timeout_s=120, sleep_s=1, terminate=terminate)
        logger.info("Ref locked")
    elif config.USRP_CONF.CLK_REF == "EXT":
        usrp.set_clock_source("external")
        _wait_for_sensor_true(usrp, "ref_locked", logger,
                              timeout_s=120, sleep_s=1, terminate=terminate)
        logger.info("Ref locked")
    else:
        usrp.set_clock_source("internal")

    if config.USRP_CONF.PPS_REF == "GPSDO":
        usrp.set_time_source("gpsdo", 0)
        _wait_for_sensor_true(usrp, "gps_locked", logger,
                              board=0, timeout_s=180, sleep_s=0.1,
                              terminate=terminate)
        _wait_for_first_pps(usrp, logger, terminate)
    elif config.USRP_CONF.PPS_REF == "EXT":
        usrp.set_time_source("external")
        _wait_for_first_pps(usrp, logger, terminate)
    # No PPS: nothing to do here; align_device_time will set_time_now(0).


def align_device_time(config, usrp, logger, terminate=None):
    if config.USRP_CONF.PPS_REF in ("GPSDO", "EXT"):
        align_device_time_to_utc(usrp, logger, terminate)
    else:
        usrp.set_time_now(uhd.types.TimeSpec(0.0))


def init_sync(config, usrp, logger, terminate=None):
    init_clock_pps_sources(config, usrp, logger, terminate)
    align_device_time(config, usrp, logger, terminate)


# Retained for any external callers that imported the old names directly.
def gpsdo_pps_lock(usrp, logger, terminate=None):
    usrp.set_time_source("gpsdo", 0)
    _wait_for_sensor_true(usrp, "gps_locked", logger,
                          board=0, timeout_s=180, sleep_s=0.1, terminate=terminate)
    _wait_for_first_pps(usrp, logger, terminate)
    align_device_time_to_utc(usrp, logger, terminate)


def ext_pps_lock(usrp, logger, terminate=None):
    usrp.set_time_source("external")
    _wait_for_first_pps(usrp, logger, terminate)
    align_device_time_to_utc(usrp, logger, terminate)
