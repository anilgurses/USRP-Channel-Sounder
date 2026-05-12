import math
import shutil
import subprocess
import time

import uhd


def align_device_time_to_utc(usrp, logger, terminate=None):
    now = time.time()
    target_utc = math.ceil(now) + 1
    # If we're within 200 ms of the next PPS edge, the USRP may not see
    # the arming command before the edge fires. Wait one tick
    # PCs are NTP synced. Therefore, 200ms is not a problem here
    if target_utc - now < 0.2:
        time.sleep(0.3)
        now = time.time()
        target_utc = math.ceil(now) + 1
    _raise_if_terminating(terminate)
    usrp.set_time_next_pps(uhd.types.TimeSpec(float(target_utc)))
    wait_s = max(0.0, target_utc - time.time()) + 0.3
    time.sleep(wait_s)
    observed = usrp.get_time_last_pps().get_real_secs()
    if abs(observed - target_utc) > 0.5:
        raise RuntimeError(
            f"UTC alignment failed: device last_pps={observed:.3f} expected~={target_utc} "
            "(check host NTP / GPSDO PPS wiring)"
        )
    logger.info(
        "device time UTC-aligned: target=%d, last_pps=%.3f, host=%.3f",
        target_utc, observed, time.time(),
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


def gpsdo_pps_lock(usrp, logger, terminate=None):
    usrp.set_time_source("gpsdo", 0)

    _wait_for_sensor_true(
        usrp,
        "gps_locked",
        logger,
        board=0,
        timeout_s=180,
        sleep_s=0.1,
        terminate=terminate,
    )

    _raise_if_terminating(terminate)
    time.sleep(1)
    align_device_time_to_utc(usrp, logger, terminate)


def ext_pps_lock(usrp, logger, terminate=None):
    usrp.set_time_source("external")

    last_pps_time = usrp.get_time_last_pps().get_real_secs()
    while last_pps_time == usrp.get_time_last_pps().get_real_secs():
        _raise_if_terminating(terminate)
        logger.info("Waiting for ref lock")
        time.sleep(0.1)

    _raise_if_terminating(terminate)
    time.sleep(1)
    align_device_time_to_utc(usrp, logger, terminate)


def init_sync(config, usrp, logger, terminate=None):
    check_host_clock_sync(logger)
    if config.USRP_CONF.CLK_REF == "GPSDO":
        usrp.set_clock_source("gpsdo")
        _wait_for_sensor_true(
            usrp,
            "ref_locked",
            logger,
            timeout_s=120,
            sleep_s=1,
            terminate=terminate,
        )
        logger.info("Ref locked")
    elif config.USRP_CONF.CLK_REF == "EXT":
        usrp.set_clock_source("external")
        _wait_for_sensor_true(
            usrp,
            "ref_locked",
            logger,
            timeout_s=120,
            sleep_s=1,
            terminate=terminate,
        )
        logger.info("Ref locked")
    else:
        usrp.set_clock_source("internal")

    if config.USRP_CONF.PPS_REF == "GPSDO":
        gpsdo_pps_lock(usrp, logger, terminate)
    elif config.USRP_CONF.PPS_REF == "EXT":
        ext_pps_lock(usrp, logger, terminate)
    else:
        usrp.set_time_now(uhd.types.TimeSpec(0.0))
