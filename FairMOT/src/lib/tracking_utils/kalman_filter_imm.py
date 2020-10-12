import numpy as np
import scipy.linalg
from kalman_filter_ca import KalmanCAFilter
from kalman_filter_cv import KalmanCVFilter


# state x (x, y, a, h, vx, vy, va, vh, ax, ay, aa, ah)
# state z (x, y, a, h)
# trans matrix [[1, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0]
#               [0, 0, 1, 0, 0, 0, dt, 0, 0, 0, 0, 0]
#               [0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
# observe matrix [[1, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0, 0]]

class KalmanIMMFilter(object):
    def __init__(self):
        ndim, dt = 4, 1
        # create different kalman filter
        filter0 = KalmanCAFilter()
        filter1 = KalmanCVFilter()
        self.filters = [filter0, filter1]
        # filter num
        self.N = 2
        self.M = np.array([[0.85, 0.15], [0.15, 0.85]])
        self.mu = np.array([0.5, 0.5])
        self.omega = np.zeros((self.N, self.N))
        self.x = np.zeros(ndim * 3)
        self.P = np.zeros((ndim * 3, ndim * 3))
        self.likelihood = np.zeros(self.N)
        self.cbar = None
        self.x_prior = None
        self.P_prior = None
        self.x_post = None
        self.P_post = None

    def initiate(self, measurement):
        """
        init the measurement
        Args:
            measurement:

        Returns:

        """
        for f in self.filters:
            _, _ = f.initiate(measurement)
        self._compute_mixing_probabilities()
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return self.x, self.P

    def update(self, measurement):
        """
        add a new measurement z to kalman filter
        Args:
            measurement:

        Returns:

        """
        # update each filter and save the likelihood
        for i, f in enumerate(self.filters):
            # need to normal the format
            f.update(f.x, f.P, measurement)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        # normalize the mu
        self.mu /= np.sum(self.mu)
        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save the posterior estimate
        self._compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        return self.x, self.P

    def predict(self):
        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)
        # compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            f.predict(xs[i], Ps[i])
        # compute mixed IMM state and covariance and save the posterior estimate
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.x.copy()
        return self.x.copy(), self.P.copy()

    def _compute_mixing_probabilities(self):
        """
        compute the mixing probability for each filter
        Returns:

        """
        self.cbar = np.dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i][j] = (self.M[i][j] * self.mu[i]) / self.cbar[j]

    def _compute_state_estimate(self):
        """
        compute the state from each filter using mu
        Returns:

        """
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x += f.x * mu

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (np.outer(y, y) + f.P)



