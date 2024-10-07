import os 
import shutil 
from datetime import datetime

# def moveVehicleLogs(logDir):
#     vehicleLogs = os.listdir(logDir + "/Results" ) 
#     radioMeasurements = os.listdir(logDir + "/measurements")

#     for log in vehicleLogs:
#         if log.endswith("_vehicleOut.txt"):
#             # decide if log date and time is in radioMeasurements 
#             logDate = datetime.strptime(str(log), '%Y-%m-%d_%H_%M_%S')
#             for meas in radioMeasurements:
#                 measDate = datetime.strptime(str(meas), '%Y-%m-%d_%H_%M_%S')
#                 if logDate == measDate:
#                     # move log to measurements folder
#                     measDate =

#                     shutil.move(logDir + "/Results/" + log, logDir + "/measurements/" + log)