# !/bin/python3

import datetime

class GpModel:

    def __init__(self, dayDivisionHour=0):
        self.divTime = datetime.time(hour=dayDivisionHour)

    def submitData(self, active, passive):
        self.activeData = active
        self.passivePassive = passive

    def createIndexTable(self, x, y):
        self.yData = self.activeData[['datetime',y]]
        print(x)
        print(self.passivePassive['timestamp'][0])
        print(self.passivePassive['timestamp'][len(self.passivePassive['timestamp'])-1])

