from kalman_filter import KalmanFilter
from kalman_filter_imm import KalmanIMMFilter
import numpy as np
np.random.seed(2020)

import matplotlib.pyplot as plt

# state x (x, y, a, h, vx, vy, va, vh, ax, ay, aa, ah)
# state z (x, y, a, h)
# trans matrix [[1, 0, 0, 0, dt, 0, 0, 0, dt*dt/2, 0, 0, 0],
#               [0, 1, 0, 0, 0, dt, 0, 0, 0, dt*dt/2, 0, 0]
#               [0, 0, 1, 0, 0, 0, dt, 0, 0, 0, dt*dt/2, 0]
#               [0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0, dt*dt/2],
#               [0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, dt]]
# observe matrix [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0, 0]]

dt, ndim = 1, 4
period_num = 50


class MotionObject(object):
    def __init__(self, x0, std_weight_position, std_weight_velocity, std_weight_acceleration):
        self.x = x0
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self._std_weight_acceleration = std_weight_acceleration
        std = [
            2 * self._std_weight_position * x0[3],
            2 * self._std_weight_position * x0[3],
            0,
            0,
            0.5 * self._std_weight_velocity * x0[3],
            0.5 * self._std_weight_velocity * x0[3],
            0,
            0,
            0.1 * self._std_weight_acceleration * x0[3],
            0.1 * self._std_weight_acceleration * x0[3],
            0,
            0]
        self.Q = np.diag(std)
        # self.Q = np.zeros((12, 12))
        print(self.Q)
        std = [
            1 * self._std_weight_position * x0[3],
            1 * self._std_weight_position * x0[3],
            0,
            0]
        self.R = np.diag(std)
        print(self.R)

        self._motion_mat = np.eye(3 * ndim, 3 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        for i in range(ndim):
            self._motion_mat[i, 2 * ndim + i] = dt * dt / 2
        for i in range(ndim, 2 * ndim):
            self._motion_mat[i, ndim + i] = dt
        print(self._motion_mat)
        self._update_mat = np.eye(ndim, 3 * ndim)

    def update(self):
        self.x = np.dot(self.x, self._motion_mat.T) + np.dot(np.random.randn(1, ndim * 3), self.Q)
        return self.x

    def sense(self):
        return np.dot(self.x, self._update_mat.T) + np.dot(np.random.randn(1, ndim), self.R)


def test_kalman_filter():
    x0 = np.array([0., 0., 1., 20., 10, 20, 0, 0, 0, 0, 0, 0])

    obj = MotionObject(x0, 1/10, 1./200, 1.0/200)
    xs, zs = [], []
    for i in range(period_num):
        x = obj.update()
        xs.append(x[0])
        z = obj.sense()
        zs.append(z[0])
    x0 = xs[-1]
    x0[8:10] = 2, 3
    obj = MotionObject(x0, 1 / 10, 1. / 200, 1.0 / 200)
    for i in range(period_num):
        x = obj.update()
        xs.append(x[0])
        z = obj.sense()
        zs.append(z[0])
    xs = np.array(xs)
    zs = np.array(zs)
    filter = KalmanIMMFilter()
    mean, cov = filter.initiate(zs[0])
    xs_filter = [xs[0, :]]
    for i in range(1, 2 * period_num):
        mean, cov = filter.predict()
        xs_filter.append(mean)
        _, _ = filter.update(zs[i])
    xs_filter = np.array(xs_filter)

    print("filter x error: {:.2f}".format(np.mean(np.abs(xs[:, 0] - xs_filter[:, 0]))))
    print("filter y error: {:.2f}".format(np.mean(np.abs(xs[:, 1] - xs_filter[:, 1]))))
    print("observe x error: {:.2f}".format(np.mean(np.abs(xs[:, 0] - zs[:, 0]))))
    print("observe y error: {:.2f}".format(np.mean(np.abs(xs[:, 1] - zs[:, 1]))))
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 0])
    plt.title("x")
    plt.subplot(2, 2, 2)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 1])
    plt.title("y")
    plt.subplot(2, 2, 3)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 2])
    plt.title("a")
    plt.subplot(2, 2, 4)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 3])
    plt.title("h")
    plt.figure(2)
    plt.subplot(3, 2, 1)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), zs[:, 0])
    plt.title('observe x')
    plt.subplot(3, 2, 2)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), zs[:, 1])
    plt.title('observe y')
    plt.subplot(3, 2, 3)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 0] - xs_filter[:, 0])
    plt.title('filter x error')
    plt.subplot(3, 2, 4)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 1] - xs_filter[:, 1])
    plt.title('filter y error')
    plt.subplot(3, 2, 5)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 0] - zs[:, 0])
    plt.title('x error')
    plt.subplot(3, 2, 6)
    plt.plot(np.linspace(1, 2 * period_num, 2 * period_num), xs[:, 1] - zs[:, 1])
    plt.title('y error')
    plt.show()


def test_kalman_ca_filter():
    pass


if __name__ == "__main__":
    test_kalman_filter()


