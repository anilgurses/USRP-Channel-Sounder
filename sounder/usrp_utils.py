import time

import uhd


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
    usrp.set_time_next_pps(uhd.types.TimeSpec(1.0))
    _raise_if_terminating(terminate)
    time.sleep(1)

    logger.info("GPS Time %s", usrp.get_mboard_sensor("gps_time").to_int())
    logger.info("USRP Time %s", usrp.get_time_last_pps().get_real_secs())


def ext_pps_lock(usrp, logger, terminate=None):
    # Can only be used on B210
    # TODO add check for this
    usrp.set_time_source("external")

    last_pps_time = usrp.get_time_last_pps().get_real_secs()
    while last_pps_time == usrp.get_time_last_pps().get_real_secs():
        _raise_if_terminating(terminate)
        logger.info("Waiting for ref lock")
        time.sleep(0.1)

    _raise_if_terminating(terminate)
    time.sleep(1)
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    _raise_if_terminating(terminate)
    time.sleep(1)

    logger.info("Ref. clock and time lock configured")
    logger.info("USRP Time %s", usrp.get_time_last_pps().get_real_secs())


def init_sync(config, usrp, logger, terminate=None):
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
