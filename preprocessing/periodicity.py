# !/bin/python3
import numpy as np

# testing signal's serial dependency
# determining periodicity using ACF (auto-correlation function)
# testing periodicity using Pearson's correlation

class Periodicity:

    def __init__(self):
        self.observations = []

    def addObservtions(self, observations):
        obs = np.array(observations, dtype='U32')
        obs_missing = np.where(obs == '-')
        obs[obs_missing] = 999999
        obs_masked = np.ma.masked_array(obs, dtype='float32')
        obs_masked[obs_missing] = np.ma.masked
        self.observations = obs_masked

        # serial-correlation function

    def serial_corr(self, lag=1):
        n = len(self.observations)
        y1 = self.observations[lag:]
        y2 = self.observations[:n - lag]
        self.scf = np.corrcoef(y1, y2, ddof=0)[0, 1]

    # auto-correlation function
    def auto_corr(self):
        self.acf = np.correlate(self.observations, self.observations, mode='same')
        print(self.acf)

    def pearson_corr(self, lag=1440):
        n = len(self.observations)/lag - 1
        self.pcf = []
        for i in range(n):
            pcf = np.corrcoef(self.observations[(i*lag):((i*lag)+lag)],
                               self.observations[((i*lag)+lag):(((i+1)*lag)+lag)], ddof=0)
            self.pcf.append(pcf)
        print(self.pcf)




