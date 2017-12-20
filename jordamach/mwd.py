import numpy as np
from math import log, sqrt
from linear_regression import Regrezio


class MWD:
    """ MWD approximation tests which for of functional structure is better for the regression model. """
    def __init__(self, x, y):
        self.r_lin = Regrezio(model='lin')
        self.r_lin.fit(x, y)
        self.r_lin.predict(x)

        self.r_log = Regrezio(model='log-log')
        self.r_log.fit(x, y)
        self.r_log.predict(self.r_log.x)

        self.w = np.array(
            [[log(y[0]) - ln_y[0]] for y, ln_y in zip(self.r_lin.y_pred, self.r_log.y_pred)]
        )
        self.x = np.array(
            [[xi[0], wi[0]] for xi, wi in zip(x, self.w)]
        )

        self.r_w = Regrezio(model='lin')
        self.r_w.fit(self.x, y)
        self.r_w.predict(self.r_w.x)

        # sum(e^2) / (n-k)
        n = self.r_w.n
        k = len(self.r_w.coefficient_vector)
        xtx_inv_last_val = self.r_w.xtx_inv[k - 1][k - 1]
        var_cov = sqrt((self.r_w.squared_errors / (n - k)) * xtx_inv_last_val)
        self.t = self.r_w.coefficient_vector[k-1] / var_cov


mwd = MWD(np.array(np.matrix('2;5;5;7;9')), np.array(np.matrix('2;4;3;5;6')))
