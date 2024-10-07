import logging
import os

class Logger():
    def __init__(self):
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        self.logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()

        self.fileHandler = logging.FileHandler("{}/{}.log".format("logs", "out"), mode="w")
        self.fileHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.fileHandler)

        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)

        self.rootLogger.setLevel(logging.DEBUG)
        
    
    def info(self,msg):
        self.rootLogger.info(msg)

    def err(self,err):
        self.rootLogger.error(err)

    def warn(self,wrn):
        self.rootLogger.warning(wrn)
