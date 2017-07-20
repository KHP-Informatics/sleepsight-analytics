import sys
import os
from tools import Logger

log_dir = ''
log_dir += sys.argv[1]

if log_dir is '':
    print('[ERROR] Please provide the log directory')
    exit()
log = Logger(log_dir, '*log_of_logs.txt', printLog=True)
log.emit('NEW STATUS REPORT', newRun=True)

filenames = os.listdir(log_dir)

sleepsightLogs = []
for filename in filenames:
    if 'sleepsight' in filename:
        print(filename)
        ssLog = Logger(log_dir, filename)
        log.emit(ssLog.getLastMessage())

