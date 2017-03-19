# !/bin/python3
import os
import json
import numpy as np
import pandas as pd
import datetime
import pickle
from HipDynamics import LookUpTable

# Participant data object

class Participant:

    @property
    def activeSensingFilenameSelector(self):
        return self.__activeSensingFilenameSelector

    @activeSensingFilenameSelector.setter
    def activeSensingFilenameSelector(self, selector):
        self.__activeSensingFilenameSelector = selector

    @property
    def metaDataFileName(self):
        return self.__metaDataFileName

    @metaDataFileName.setter
    def metaDataFileName(self, filename):
        self.__metaDataFileName = filename

    @property
    def passiveData(self):
        return self.__passiveData

    @passiveData.setter
    def passiveData(self, data):
        self.__passiveData = data

    @property
    def activeData(self):
        return self.__activeData

    @activeData.setter
    def activeData(self, data):
        self.__activeData = data

    @property
    def passiveSensors(self):
        psWithoutTimeStamp = []
        for ps in self.__passiveSensors:
            if ps not in 'timestamp':
                psWithoutTimeStamp.append(ps)
        return psWithoutTimeStamp

    @passiveSensors.setter
    def passiveSensors(self, arr):
        self.__passiveSensors = arr

    @property
    def missingness(self):
        return self.__missingness

    @missingness.setter
    def missingness(self, m):
        self.__missingness = m

    @property
    def periodicity(self):
        return self.__periodicity

    @periodicity.setter
    def periodicity(self, p):
        self.__periodicity = p

    def __init__(self, id, path=''):
        self.id = self.formatID(id)
        self.path = path
        self.pipelineStatus = {
            'active data': False,
            'passive data': False,
            'trim data': False,
            'merging data': False,
            'missingness': False,
            'imputation': False,
            'periodicity': False,
            'GP model gen.': False,
            'anomaly detect.': False,
            'association': False
        }
        self.metaDataFileName = ''
        self.info = dict()


    def formatID(self, id_int):
        id_str = str(id_int)
        if len(id_str) < 2:
            id_str = '0' + id_str
        return id_str

    def load(self):
        print('[Participant] Loading data...')
        files = os.listdir(self.path)
        filtered_files = [file for file in files if self.id in file]
        (activeSensFiles, passiveSensFiles) = self.splitFilesIntoActiveAndPassive(filtered_files)
        if not self.containsParticipantObj(passiveSensFiles):
            self.loadPassiveData(passiveSensFiles)
            self.loadActiveData(activeSensFiles)
            if self.metaDataFileName is not '':
                self.loadMetaData(self.metaDataFileName)
            self.saveSnapshot()
        else:
            self.loadSnapshot()

    def splitFilesIntoActiveAndPassive(self, files):
        active = []
        passive = []
        for file in files:
            if self.activeSensingFilenameSelector in file:
                active.append(file)
            else:
                #reserved for meta data
                if 'meta' not in file:
                    passive.append(file)
        return (active, passive)

    def loadPassiveData(self, filenames):
        print('[Participant] No existing x_Participant.pkl file was found. ' +
              'A merged LookUpTable is genereated from found source files.\n' +
              '       [NOTE] Depending on the size of your datset this may take ' +
              'several minutes to hours.')
        pTable = self.generatePassiveDataTable(filenames)
        self.updatePipelineStatusForTask('passive data')
        self.passiveData = self.transformTableIntoNumpyArray(pTable)
        self.updatePipelineStatusForTask('merging data')

    def containsParticipantObj(self, filenames):
        for filename in filenames:
            if 'participant.pkl' in filename:
                return True
        return False

    def generatePassiveDataTable(self, filenames):
        individual_tables = []
        for filename in filenames:
            file_path = self.path + filename
            input_data = np.genfromtxt(file_path, delimiter=',')
            table = self.generateDataTable(input_data, self.formatSensorName(filename))
            individual_tables.append(table)

        heartTable_index = [i for i in range(len(filenames)) if 'heart' in filenames[i]][0]
        time_arr = np.sort(individual_tables[heartTable_index].table['timestamp'])
        p_table = self.generateTimeDataTable(time_arr)

        for i in range(len(individual_tables)):
            p_table.annotateWith(individual_tables[i])
        return p_table

    def generateDataTable(self, input_data, sensor_name):
        lt = LookUpTable()
        lt.mapping = [{'timestamp': []}, {sensor_name: []}]
        for row in input_data:
            row = list(row)
            try:
                row[0] = datetime.datetime.fromtimestamp(int(row[0])).strftime('%Y-%m-%d %H:%M')
                lt.add(row)
            except ValueError:
                print('[WARN] During data table generation of {} a ValueError occurred.'.format(sensor_name))
        return lt

    def formatSensorName(self, filename):
        formatted = filename[3:-4]
        if 'milli' in formatted:
            formatted = formatted[:-6]
        return formatted

    def generateTimeDataTable(self, time_arr):
        t_first = time_arr[0]
        t_last = time_arr[(len(time_arr) - 1)]
        lt = LookUpTable()
        lt.mapping = [{'timestamp': []}]
        t_min = datetime.timedelta(minutes=1)
        t_iter = datetime.datetime.strptime(t_first, '%Y-%m-%d %H:%M')
        t_end = datetime.datetime.strptime(t_last, '%Y-%m-%d %H:%M')
        while t_iter <= t_end:
            lt.add([t_iter.strftime('%Y-%m-%d %H:%M')])
            t_iter = t_iter + t_min
        return lt

    def transformTableIntoNumpyArray(self, table):
        keys = sorted(table.table.keys())
        self.passiveSensors = keys
        col_names = ['timestamp']
        ts = np.array(table.table['timestamp'])
        for key in keys:
            if str(key) not in 'timestamp':
                col_names.append(key)
                ts = np.column_stack((ts, table.table[key]))
        pd_ts = pd.DataFrame(ts, columns=col_names)
        return pd_ts

    def loadActiveData(self, filenames):
        if '{}_symptoms_diary_tabluar.csv'.format(self.id) in filenames:
            self.loadActiveDataSymptoms()
            self.updatePipelineStatusForTask('active data')
        else:
            print('[WARN] No {}_symptoms_diary_tabluar.csv was found. Please load active data manually.'.format(self.id))

    def loadActiveDataSymptoms(self):
        file_path = self.path + '{}_symptoms_diary_tabluar.csv'.format(self.id)
        input_data = np.genfromtxt(file_path, delimiter=',', dtype="U")
        data = self.formatSymptomsData(input_data)
        self.activeData = self.scoreSymptomsData(data)

    def formatSymptomsData(self, input_data):
        cols = self.formatRawSymptomsCols(input_data[0])
        data = pd.DataFrame(input_data[1:], columns=cols)
        data['timestamp'] = self.formatTimestampToDateTime(data['timestamp'])
        for i in range(1, len(cols)):
            data[cols[i]] = pd.to_numeric(data[cols[i]], downcast='signed', errors='coerce')
        data.sort_values(by=['timestamp'], ascending=True, inplace=True)
        cols = list(data.columns.values)
        cols_sorted = np.concatenate((['timestamp'], sorted(cols[1:])))
        data = data[cols_sorted]
        return data

    def formatRawSymptomsCols(self, s_cols):
        def formatRawNames(col):
            c = col[44:50]
            c = c.split('_')[0]
            return c
        cols = [formatRawNames(col) for col in s_cols]
        cols[0] = 'timestamp'
        return cols

    def formatTimestampToDateTime(self, arr):
        def toDateTime(val):
            v = val.split('.')[0]
            return datetime.datetime.fromtimestamp(int(v)).strftime('%Y-%m-%d %H:%M')
        a = [toDateTime(val) for val in arr]
        return a

    def scoreSymptomsData(self, data):
        cols = ['datetime', 'hopelessness', 'depression', 'auditory halucination', 'visual halucination',
                'anxiety', 'grandiosity', 'suspiciousness', 'total']
        q1_data = data.filter(regex="q1")
        q2_data = data.filter(regex="q2")
        q3_data = data.filter(regex="q3")
        q3a_data = q3_data[["q3i", "q3ia", "q3ib", "q3ic"]]
        q3b_data = q3_data[["q3ii", "q3iia", "q3iib", "q3iic"]]
        q4_data = data.filter(regex="q4")
        q5_data = data.filter(regex="q5")
        q6_data = data.filter(regex="q6")

        q_data = [q1_data, q2_data, q3a_data, q3b_data, q4_data, q5_data, q6_data]

        scores = [data['timestamp'].values.tolist()]
        for j in range(len(q_data)):
            row_means = []
            for i in range(len(data)):
                row_val = q_data[j][i:(i+1)].values.tolist()[0]
                if j == 5:
                    row_v_f = self.formatSymptomsRowScores(row_val, exception_grandiosity=True)
                else:
                    row_v_f = self.formatSymptomsRowScores(row_val)
                row_mean = np.mean(row_v_f)
                row_means.append(row_mean)
            scores.append(row_means)

        pd_s = pd.DataFrame(scores).T
        pd_s[8] = pd_s[[1,2,3,4,5,6]].sum(axis=1)
        pd_s.columns = cols
        return pd_s


    def formatSymptomsRowScores(self, val, exception_grandiosity=False):
        v = []
        for i in range(len(val)):
            if val[i] == 0:
                v.append(1)
            elif val[i] == -1:
                pass
            elif exception_grandiosity and val[i] < 5:
                v.append(1)
            elif exception_grandiosity and val[i] == 5:
                v.append(2)
            elif exception_grandiosity and val[i] == 6:
                v.append(3)
            elif exception_grandiosity and val[i] == 7:
                v.append(4)
            else:
                v.append(val[i])
        return v

    def loadMetaData(self, filename, selector=''):
        if selector == '':
            selector = self.id
        path = self.path + filename
        print(path)
        with open(path) as data_file:
            data = json.load(data_file)
        self.info = data[selector]

    def trimData(self, startDate, endDate='', duration=0):
        start, end = self.determineStartAndEndDates(startDate, endDate, duration)
        trimmedPassiveIdx = self.getTrimmedPassiveIndecies(start, end)
        self.passiveData = self.passiveData.drop(self.passiveData.index[trimmedPassiveIdx])
        trimmedActiveIdx = self.getTrimmedActiveIndecies(start, end)
        self.activeData = self.activeData.drop(self.activeData.index[trimmedActiveIdx])

    def determineStartAndEndDates(self, startDate, endDate='', duration=0):
        start = datetime.datetime.strptime(startDate, '%d/%m/%Y')
        if len(endDate) > 0:
            end = datetime.datetime.strptime(endDate, '%d/%m/%Y')
        elif duration > 0:
            end = start + datetime.timedelta(days=(duration + 1))
        else:
            end = datetime.datetime.strptime(self.passiveData['timestamp'][len(self.passiveData['timestamp'])-1], '%Y-%m-%d %H:%M')
        return (start, end)

    def getTrimmedPassiveIndecies(self, start, end):
        idxs = []
        for i in range(0, len(self.passiveData)):
            dateOfInterest = datetime.datetime.strptime(self.passiveData['timestamp'][i], '%Y-%m-%d %H:%M')
            if dateOfInterest < start or end < dateOfInterest:
                idxs.append(i)
        return idxs

    def getTrimmedActiveIndecies(self, start, end):
        idxs = []
        for i in range(0, len(self.activeData)):
            dateOfInterest = datetime.datetime.strptime(self.activeData['datetime'][i], '%Y-%m-%d %H:%M')
            if dateOfInterest < start or end < dateOfInterest:
                idxs.append(i)
        return idxs


    def getPassiveDataColumn(self, col=''):
        if col is '':
            print('[ERR] Supply one of your column names.\n      Choose from {}.'.format(str(self.passiveCols)))
            exit()
        else:
            return self.passiveData[col]

    def setPassiveDataColumn(self, data, col=''):
        if col is '':
            print('[ERR] Supply one of your column names.\n      Choose from {}.'.format(str(self.passiveCols)))
            exit()
        else:
            self.passiveData[col] = np.array(data)

    def updatePipelineStatusForTask(self, taskName, status=True):
        keys = self.pipelineStatus.keys()
        if taskName in keys:
            self.pipelineStatus[taskName] = status

    def isPipelineTaskCompleted(self, taskName):
        keys = self.pipelineStatus.keys()
        if taskName in keys:
            return self.pipelineStatus[taskName]
        else:
            print('[ERR] Requested task <{}> is unknown.'.format(taskName))
            exit()

    def saveSnapshot(self, path=''):
        if path not in '':
            self.path = path
        output_filen = self.path + self.id + "_participant.pkl"
        with open(output_filen, 'wb') as output:
            pickle._dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def loadSnapshot(self):
        filename = self.path + self.id + "_participant.pkl"
        with open(filename, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
        self.__dict__.update(tmp_dict)

    def __str__(self):
        classInfo = '\nClass {} loads all relevant sensor data, formats and cashes it according to the state of ' \
               'the pipeline.\n\n'
        participantInfo = ''
        if len(self.info) > 0:
            participantInfo += 'Participant Info:\n' \
                              '   ID:               {}\n' \
                              '   Age:              {} years\n' \
                              '   Gender:           {}\n'\
                              '   Start date:       {}\n' \
                              '   Illness duration: {} years\n' \
                              '   PANSS score:      {}\n' \
                              '   Medication:       {}\n\n'.format(self.info['id'],
                                                                   self.info['age'],
                                                                   self.info['gender'],
                                                                   self.info['startDate'],
                                                                   self.info['durationIllness'],
                                                                   self.info['PANSS']['total'],
                                                                   str(self.info['medication']))
        pipelineInfo = 'Current Pipeline Status of Participant {}:\n\n' \
               '>> passive data[{}] & active data[{}]\n    |\n>> trim[{}]\n    |\n' \
               '>> merging[{}] --------|\n    |                    |' \
               '\n>> imputation[{}]    missingness[{}]\n    |\n>> periodicity[{}]\n    |\n>> GP model gen.[{}]\n' \
               '    |\n>> Anomaly detect.[{}]'.format(type(self), self.id,
                                                    self.pipelineStatus['passive data'],
                                                    self.pipelineStatus['active data'],
                                                    self.pipelineStatus['trim data'],
                                                    self.pipelineStatus['merging data'],
                                                    self.pipelineStatus['imputation'],
                                                    self.pipelineStatus['missingness'],
                                                    self.pipelineStatus['periodicity'],
                                                    self.pipelineStatus['GP model gen.'],
                                                    self.pipelineStatus['Anomaly detect.'],
                                                    self.pipelineStatus['Association']
                                                )
        rendered = classInfo + participantInfo + pipelineInfo
        return rendered
