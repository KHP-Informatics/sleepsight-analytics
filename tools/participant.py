# !/bin/python3
import os
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
        return self.__passiveSensors

    @passiveSensors.setter
    def passiveSensors(self, arr):
        self.__passiveSensors = arr

    @property
    def missingness(self):
        return self.__missingness

    @missingness.setter
    def missingness(self, m):
        self.__missingness = m

    def __init__(self, id, path=''):
        self.id = self.formatID(id)
        self.path = path
        self.pipelineStatus = {
            'active data': False,
            'passive data': False,
            'merging data': False,
            'missingness': False,
            'imputation': False,
            'periodicity': False,
            'GP model gen.': False,
            'Anomaly detect.': False,
            'Association': False
        }


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
        self.loadPassiveData(passiveSensFiles)
        self.loadActiveData(activeSensFiles)

    def splitFilesIntoActiveAndPassive(self, files):
        active = []
        passive = []
        for file in files:
            if self.activeSensingFilenameSelector in file:
                active.append(file)
            else:
                passive.append(file)
        return (active, passive)

    def loadPassiveData(self, filenames):
        if not self.containsParticipantObj(filenames):
            print('[Participant] No existing x_Participant.pkl file was found. ' +
                  'A merged LookUpTable is genereated from found source files.\n' +
                  '       [NOTE] Depending on the size of your datset this may take ' +
                  'several minutes to hours.')
            pTable = self.generatePassiveDataTable(filenames)
            self.updatePipelineStatusForTask('passive data')
            self.passiveData = self.transformTableIntoNumpyArray(pTable)
            self.updatePipelineStatusForTask('merging data')
            self.saveSnapshot()
        else:
            self.loadSnapshot()

    def loadActiveData(self, filenames):
        #self.updatePipelineStatusForTask('active data')
        pass

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

    def saveSnapshot(self):
        output_filen = self.path + self.id + "_participant.pkl"
        with open(output_filen, 'wb') as output:
            pickle._dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def loadSnapshot(self):
        filename = self.path + self.id + "_participant.pkl"
        with open(filename, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
        self.__dict__.update(tmp_dict)

    def __str__(self):
        return '\nClass {} loads all relevant sensor data, formats and cashes it according to the state of ' \
               'the pipeline.\n\nCurrent Pipeline Status of Participant {}:\n\n' \
               '>> passive data[{}] & active data[{}]\n    |\n>> merging[{}] --------|\n    |                    |' \
               '\n>> imputation[{}]    missingness[{}]\n    |\n>> periodicity[{}]\n    |\n>> GP model gen.[{}]\n' \
               '    |\n>> Anomaly detect.[{}]'.format(type(self), self.id,
                                                    self.pipelineStatus['passive data'],
                                                    self.pipelineStatus['active data'],
                                                    self.pipelineStatus['merging data'],
                                                    self.pipelineStatus['imputation'],
                                                    self.pipelineStatus['missingness'],
                                                    self.pipelineStatus['periodicity'],
                                                    self.pipelineStatus['GP model gen.'],
                                                    self.pipelineStatus['Anomaly detect.'],
                                                    self.pipelineStatus['Association']
                                                )
