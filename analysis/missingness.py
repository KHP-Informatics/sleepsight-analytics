# !/bin/python3
import numpy as np
import pandas as pd
import datetime
from tools import TreeLeaf, TreeNode

# Module to categorise values into:
# 1. Not missing
# 2. Missing, due to PACKAGE LOSS
# 3. Missing, due to NOT WEARING THE DEVICE
# 4. Missing, due to NOT CHARGING THE DEVICE
# Note: the decision tree is specific to the data collected during the SleepSight pilot study


# DT
# If all intra FB values are missing: either not worn
# If battery =<= 1 not charged
# If phone sensors + (non intra) wearables missing for more than 6 minutes = package loss

class MissingnessDT2:

    def __init__(self, passiveData, activeData):
        self.passive = passiveData
        self.active = activeData

    def constructDecisionTree(self):

        def evalBattery(data):
            batteryEvalStart = data[['timestamp', 'battery']].values[0]
            batteryEvalEnd = data[['timestamp', 'battery']].values[4]
            if batteryEvalStart[1] in '-' or float(batteryEvalStart[1]) > 1:
                return (False, data)
            else:
                dateStart = datetime.datetime.strptime(batteryEvalStart[0], '%Y-%m-%d %H:%M')
                dateEnd = datetime.datetime.strptime(batteryEvalEnd[0], '%Y-%m-%d %H:%M')
                elapsedTime = dateEnd - dateStart
                elapsedMin = elapsedTime.total_seconds() / 60
                return (True, batteryEvalStart[0])

        def evalHeartRate(data):
            heartEval = data[['timestamp', 'heart_Minutes']].values
            if heartEval[0][1] not in '-':
                return (False, data)
            else:
                hasNotValue = True
                for i in range(1, len(heartEval)):
                    if heartEval[i][1] not in '-':
                        hasNotValue = False
                        if not hasNotValue:
                            break
                return (hasNotValue, heartEval[0][0])

        def evalTransmissionFailure(data):
            tfTime = data['timestamp'].values
            tfEval = data[['steps', 'light', 'FAM', 'LAM', 'VAM', 'awake_count']].values
            tfEvalFlat = [val for sublist in tfEval for val in sublist]
            hasNotValue = True
            for i in range(0, len(tfEvalFlat)):
                if tfEvalFlat[i] not in '-':
                    hasNotValue = False
                    if not hasNotValue:
                        break
            return (hasNotValue, tfTime[0])

        def finaLeafPass(data):
            time = data.values[0][0]
            return (True, time)

        def channelThrough(d):
            return (True, d)

        def simFailure(d):
            return (False, d)

        leaf0 = TreeLeaf(name='Not Charged', evalMethod=evalBattery)
        leaf1 = TreeLeaf(name='Not Worn', evalMethod=evalHeartRate)
        leaf2 = TreeLeaf(name='Transmission Failure', evalMethod=evalTransmissionFailure)
        leaf3 = TreeLeaf(name='No Missingness', evalMethod=finaLeafPass)
        node2 = TreeNode(name='Leaf2<>Leaf3', children=[leaf2, leaf3], evalMethod=channelThrough)
        node1 = TreeNode(name='Leaf1<>Node2', children=[leaf1, node2], evalMethod=channelThrough)
        self.root = TreeNode(name='Root[Leaf0<>Node1]', children=[leaf0, node1], evalMethod=channelThrough)

    def run(self):
        for i in range(self.passive.index[0], self.passive.index[round(len(self.passive.index)-10)]):
            self.root.invoke(self.passive.loc[i:(i+10)])
        self.result = self.root.retrieveLeaves()

    def testing(self):
        pass

    def formatMissingness(self):
        self.missingness = {'count':dict(), 'daily':dict()}
        for category in self.result:
            self.missingness['count'][category.name] = len(category.result)
            self.missingness['daily'][category.name] = self.countDaily(category)
        print(self.missingness)

    def countDaily(self, category):
        idxEnd = len(self.passive['timestamp'].values)-1
        dayEnd = datetime.datetime.strptime(self.passive['timestamp'].values[idxEnd], '%Y-%m-%d %H:%M')
        dayStart = datetime.datetime.strptime(self.passive['timestamp'].values[0], '%Y-%m-%d %H:%M')
        delta = dayEnd - dayStart
        dailyCount = [0]*delta.days
        if len(category.result) > 0:
            evalDay = datetime.datetime.strptime(category.result[0], '%Y-%m-%d %H:%M')
            dayIdx = 0
            for dateTime in category.result:
                currentDay = datetime.datetime.strptime(dateTime, '%Y-%m-%d %H:%M')
                if evalDay.date() == currentDay.date():
                    dailyCount[dayIdx] += 1
                else:
                    evalDay = currentDay
                    dayIdx +=1
                    if dayIdx < len(dailyCount):
                        dailyCount[dayIdx] += 1
                    else:
                        break
        return dailyCount





    def __str__(self):
        return self.root.__str__()








class MissingnessDT:

    @property
    def missingness(self):
        self.__missingness = {
            'total': self.minutesTotal,
            'notWorn': self.minutesNotWorn,
            'notCharged': self.minutesNotCharged,
            'notMissing': self.minutesNoMissingness
        }
        return self.__missingness

    @missingness.setter
    def missingness(self, m):
        self.minutesTotal = m['total']
        self.minutesNotWorn = m['notWorn']
        self.minutesNotCharged = m['notCharged']
        self.minutesNoMissingness = m['notMissing']

    def __init__(self):
        self.phone_sensors = []
        self.wearable_sensors = []
        self.wearable_sensors_intra = []
        self.wearable_sensors_non_intra = []
        self.minutesTotal = 0
        self.minutesNoMissingness = 0
        self.minutesNotWorn = 0
        self.minutesPackageLoss = 0
        self.minutesNotCharged = 0

    def setSensors(self, sensors):
        for sensor in sensors:
            if sensor not in 'timestamp':
                if sensor in ['light', 'battery', 'accelerometer']:
                    self.phone_sensors.append(sensor)
                else:
                    self.wearable_sensors.append(sensor)
                    if 'intra' in sensor:
                        self.wearable_sensors_intra.append(sensor)
                    else:
                        self.wearable_sensors_non_intra.append(sensor)


    def setDataset(self, data):
        self.data = data

    def computeMissingness(self):
        self.hasRequiredInformation()
        self.minutesTotal = len(self.data)
        for i in range((self.minutesTotal-12)):
            self.runDecisionTree(self.data[i:(i+12)])

    def hasRequiredInformation(self):
        if len(self.data) > 0 and len(self.phone_sensors) > 0 and len(self.wearable_sensors) > 0:
            return True
        print('[ERR] You need to setSensors() and setDataset() first.')
        exit()

    def runDecisionTree(self, rows):
        if self.DtLevelOneANotCharged(rows['battery']):
            self.minutesNotCharged += 1
        elif self.DtLevelOneBNotWorn(rows['heart_Minutes']):
            self.minutesNotWorn += 1
        #elif self.DtLevelOneCPackageLoss(rows):
        #    self.minutesPackageLoss += 1
        else:
            self.minutesNoMissingness += 1

    def DtLevelOneANotCharged(self, batteryStatus):
        if list(batteryStatus[0:1])[0] is not '-':
            battInt = float(batteryStatus[0:1])
            if battInt <= 1:
                return True
        return False

    def DtLevelOneBNotWorn(self, notWornArr):
        if list(notWornArr)[0] is '-':
            return True
        return False

    def DtLevelOneCPackageLoss(self, rows, excludeSensors=['light', 'min_sleep', 'sleep_Minutes']):
        sensors = self.wearable_sensors_non_intra + self.phone_sensors
        intermittedSensors = np.setdiff1d(sensors, excludeSensors)
        iValsT = np.array(rows[intermittedSensors]).transpose()
        iValsEval = []
        for colIdx in range(len(iValsT)):
            iValsEval.append(self.containsValues(iValsT[colIdx]))
        unique, counts = np.unique(iValsEval, return_counts=True)
        nOfPresentandLostValues = dict(zip(unique, counts))
        if False in nOfPresentandLostValues.keys():
            if nOfPresentandLostValues[False] > 0:
                return True
        return False

    def containsValues(self, listVals, NA='-'):
        for val in listVals:
            if val not in NA:
                return True
        return False

    def __str__(self):
        rendered = '\nMissingness:\n\n'
        rendered += 'Not worn:     {}hrs ({}%)\n'.format(round(self.minutesNotWorn/60, 2),
                                                   round((self.minutesNotWorn/self.minutesTotal*100), 0))
        rendered += 'Not charged:  {}hrs ({}%)\n'.format(round(self.minutesNotCharged/60, 2),
                                                   round((self.minutesNotCharged / self.minutesTotal * 100), 0))
        rendered += '\nNot missing:  {}hrs ({}%)\n'.format(round(self.minutesNoMissingness / 60, 2),
                                                         round((self.minutesNoMissingness / self.minutesTotal * 100), 0))
        return rendered