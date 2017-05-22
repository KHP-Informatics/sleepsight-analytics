# !/bin/python3

import datetime

class GpModel:

    def __init__(self, xFeatures, yFeature, dayDivisionHour=0):
        self.divTime = datetime.time(hour=dayDivisionHour)
        self.xFeatures = xFeatures
        self.yFeature = yFeature

    def submitData(self, active, passive):
        self.activeData = active
        self.passiveData = passive
        self.yData = self.activeData[['datetime', self.yFeature]]
        xSelection = ['timestamp'] + self.xFeatures
        self.xData = self.passiveData[xSelection]

    def createIndexTable(self):
        self.indexDict = []
        self.extractDateIdxsFromYData()
        self.extractDateIdxsFromXDataBasedOnY()

    def extractDateIdxsFromYData(self):
        for i in range(len(self.yData)):
            entry = {'index': i}
            entry['y'] = float(self.yData[i:i+1][self.yFeature])
            startDate, endDate = self.determineDatesFromYData(i)
            entry['dateStart'] = startDate
            entry['dateEnd'] = endDate
            self.indexDict.append(entry)

    def determineDatesFromYData(self, index):
        dt_str = list(self.activeData[index:(index+1)]['datetime'])[0]
        dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
        dtEnd = dt.replace(hour=self.divTime.hour, minute=0)
        tDay = datetime.timedelta(days=1)
        tMin = datetime.timedelta(minutes=1)
        dtStart= dtEnd - tDay + tMin
        return (dtStart, dtEnd)

    def extractDateIdxsFromXDataBasedOnY(self):
        idxStart = 0
        idxEnd = 0
        currentTableIndex = 0
        for i in range(len(self.xData)):
            dateStart = self.indexDict[currentTableIndex]['dateStart']
            dateEnd = self.indexDict[currentTableIndex]['dateEnd']
            dateXDataStr = list(self.xData[i:(i + 1)]['timestamp'])[0]
            dateXData = datetime.datetime.strptime(dateXDataStr, '%Y-%m-%d %H:%M')
            if dateXData <= dateStart and dateXData < dateEnd:
                idxStart = i
            if dateXData <= dateEnd:
                idxEnd = i
            if dateXData == dateEnd or i == (len(self.xData) - 1):
                self.indexDict[currentTableIndex]['indexStart'] = idxStart
                self.indexDict[currentTableIndex]['indexEnd'] = idxEnd
                currentTableIndex += 1

