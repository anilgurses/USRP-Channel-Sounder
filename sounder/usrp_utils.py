import uhd
import time


def createMultiUSRP(config):
    dev_str = f"num_recv_frames={config.USRP_CONF.NUM_FRAMES},num_send_frames={config.USRP_CONF.NUM_FRAMES}"
    dev_str += f",serial={config.USRP_CONF.SERIAL}" if config.USRP_CONF.SERIAL else ""
    usrp = uhd.usrp.MultiUSRP(dev_str)
    return usrp


def gpsdo_pps_lock(usrp, logger):
    usrp.set_time_source("gpsdo", 0)

    while not usrp.get_mboard_sensor("gps_locked", 0).to_bool():
        logger.info("Waiting for GPS lock")
        time.sleep(0.1)

    time.sleep(1)
    usrp.set_time_next_pps(uhd.types.TimeSpec(1.0))
    time.sleep(1)

    logger.info(f"GPS Time {usrp.get_mboard_sensor('gps_time').to_int()}")
    logger.info(f"USRP Time {usrp.get_time_last_pps().get_real_secs()}")


def ext_pps_lock(usrp, logger):
    # Can only be used on B210
    # TODO add check for this
    usrp.set_time_source("external")

    last_pps_time = usrp.get_time_last_pps().get_real_secs()
    while last_pps_time != usrp.get_time_last_pps().get_real_secs():
        logger.info("Waiting for ref lock")
        time.sleep(0.1)

    time.sleep(1)
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1)

    logger.info("Ref. clock and time lock configured")
    logger.info(f"USRP Time {usrp.get_time_last_pps().get_real_secs()}")


def init_sync(config, usrp, logger):
    if config.USRP_CONF.CLK_REF == "GPSDO":
        usrp.set_clock_source("gpsdo")
        while not usrp.get_mboard_sensor("ref_locked").to_bool():
            logger.info("Waiting for ref lock")
            time.sleep(1)
        logger.info("Ref locked")
    elif config.USRP_CONF.CLK_REF == "EXT":
        usrp.set_clock_source("external")
        while not usrp.get_mboard_sensor("ref_locked").to_bool():
            logger.info("Waiting for ref lock")
            time.sleep(1)
        logger.info("Ref locked")
    else:
        usrp.set_clock_source("internal")

    if config.USRP_CONF.PPS_REF == "GPSDO":
        gpsdo_pps_lock(usrp, logger)
    elif config.USRP_CONF.PPS_REF == "EXT":
        ext_pps_lock(usrp, logger)
    else:
        usrp.set_time_now(uhd.types.TimeSpec(0.0))
