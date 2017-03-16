# !/bin/python3
import numpy as np
import matplotlib.pyplot as pl
from pykalman import KalmanFilter

# The module focuses on regularising time intervals to once per minute,
# including imputation of missing values, using the Kalman filter.

class KalmanImputation:

    @property
    def imputedObservations(self):
        return self.predicted_states[:, 0]

    def __init__(self, transition_matrices=[], transition_covariance=[]):
        if len(transition_matrices) is 0:
            transition_matrices = np.array([[1, 1], [0, 1]])
        if len(transition_covariance) is 0:
            transition_covariance = 0.01 * np.eye(2)
        # create a Kalman Filter by hinting at the size of the state and observation
        # space.  If you already have good guesses for the initial parameters, put them
        # in here.  The Kalman Filter will try to learn the values of all variables.
        self.kf = KalmanFilter(transition_matrices=transition_matrices, transition_covariance=transition_covariance)
        self.observations = []
        self.means_covs = []
        self.predicted_states = []
        self.fittedKalmanFilter = False

    def addObservtions(self, observations):
        obs = np.array(observations, dtype='U32')
        obs_missing = np.where(obs == '-')
        obs[obs_missing] = 999999
        obs_masked = np.ma.masked_array(obs, dtype='float32')
        obs_masked[obs_missing] = np.ma.masked
        self.observations = obs_masked

    def fit(self, n_iterations=10):
        if not self.fittedKalmanFilter:
            self.kf.em(self.observations, n_iter=n_iterations)
            self.fittedKalmanFilter = True


    def smooth(self):
        self.means_covs = self.kf.smooth(self.observations)
        self.predicted_states = self.means_covs[0]

    def limitToPositiveVals(self):
        negs = np.where(self.predicted_states[:, 0] < 0)
        if len(negs) > 0:
            self.predicted_states[negs][0] = 0

    def plot(self):
        # Plot lines for the observations without noise, the estimated position of the
        # target before fitting, and the estimated position after fitting.
        x = np.array(range(len(self.observations)))
        pl.figure(figsize=(16, 6))
        obs_scatter = pl.scatter(x, self.observations, marker='x', color='b',
                                 label='observations')
        position_line = pl.plot(x, self.predicted_states[:, 0],
                                linestyle='-', color='r',
                                label='position est.')
        #velocity_line = pl.plot(x, self.predicted_states[:, 1],
        #                        linestyle='-', marker='o', color='g',
        #                        label='velocity est.')
        pl.legend(loc='lower right')
        pl.xlim(xmin=0, xmax=x.max())
        pl.xlabel('time')
        pl.show()


