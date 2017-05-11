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

class MissingnessDT:

    def __init__(self, passiveData, activeDataSymptom, activeDataSleep, startDate):
        self.passive = passiveData
        self.activeSymptom = activeDataSymptom
        self.activeSleep = activeDataSleep
        self.startDate = datetime.datetime.strptime(startDate, '%d/%m/%Y')

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


    def formatMissingness(self):
        self.missingness = {'count':dict(), 'daily':dict()}
        for category in self.result:
            self.missingness['count'][category.name] = len(category.result)
            self.missingness['daily'][category.name] = self.countDailyTreeLeaf(category)
        self.missingness['count']['symptom'] = len(self.activeSymptom['datetime'].values)
        self.missingness['daily']['symptom'] = self.countDailyActive(self.activeSymptom['datetime'].values)
        self.missingness['count']['sleep'] = len(self.activeSleep['datetime'].values)
        self.missingness['daily']['sleep'] = self.countDailyActive(self.activeSleep['datetime'].values)
        print(self.missingness)

    def countDailyTreeLeaf(self, category):
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

    def countDailyActive(self, dates):
        dates = self.activeSymptom['datetime'].values
        dailyCount = [0]*56
        dayIdx = 0
        evalDay = self.startDate
        for dateTime in dates:
            currentDay = datetime.datetime.strptime(dateTime, '%Y-%m-%d %H:%M')
            if evalDay.date() == currentDay.date():
                dailyCount[dayIdx] += 1
            else:
                evalDay = currentDay
                dayIdx += 1
                if dayIdx < len(dailyCount):
                    dailyCount[dayIdx] += 1
                else:
                    break
        return dailyCount

    def __str__(self):
        rendered = '\nMissingness Summary (Count)\n'
        for key in self.missingness['count'].keys():
            rendered += '{}: {}\n'.format(str.capitalize(key), self.missingness['count'][key])
        return rendered
