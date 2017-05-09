# !/bin/python3
import numpy as np
import pandas as pd
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

    def __init__(self):
        pass



def rootMethod(b):
    return b

def lower10(val):
    if val < 10:
        return (True, val)
    return (False, val)

def greater10(val):
    if val > 10:
        return (True, val)
    return (False, val)

def equal10(val):
    if val == 10:
        return (True, val)
    return (False, val)

def node1M(val):
    if val >= 10:
        return (True, val)
    return (False, val)

leaf1 = TreeLeaf(name="Lower", evalMethod=lower10)
leaf2 = TreeLeaf(name="Greater", evalMethod=greater10)
leaf3 = TreeLeaf(name="Equal", evalMethod=equal10)
node1 = TreeNode(name='node1', children=[leaf2, leaf3], evalMethod=node1M)
root = TreeNode(name='root', children=[leaf1, node1], evalMethod=rootMethod)

data = [1,2,3,3,10,7,6,45,213,235,46,54,6,34,234,23,10,4,3,4,55,6,47,2]

for d in data:
    root.invoke(d)
print(root)

l = root.retrieveLeaves()
print(l)



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