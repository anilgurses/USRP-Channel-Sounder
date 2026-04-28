import logging
import os


class Logger:
    def __init__(self, name="sounder", log_dir="logs", level=logging.DEBUG, console=True):
        os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s"
        )

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

        file_handler = logging.FileHandler(os.path.join(log_dir, "out.log"), mode="w")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def err(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
