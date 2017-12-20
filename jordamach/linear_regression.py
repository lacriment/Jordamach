from math import log
import numpy as np


class Regrezio:
    """Base class for linear regression models"""

    def __init__(self, model='lin'):
        # set the regression model
        self.model = model.lower()

    def fit(self, x, y):
        """ Fits values to linear regression model and calculates
            coefficients and intercept """
        self.y = np.array(y)
        self.x = np.array(x)

        # Calculate the lns of the values for function models
        if self.model == 'log-lin':
            self.y = np.array([[log(i[0])] for i in self.y])
        elif self.model == 'log-log':
            self.y = np.array([[log(i[0])] for i in self.y])
            self.x = np.array([[log(cell) for cell in row] for row in self.x])
        elif self.model == 'lin-log':
            self.x = np.array([[log(cell) for cell in row] for row in self.x])
        elif self.model == 'quadratic':
            self.x = np.array([[xij[0], xij[0] ** 2] for xij in self.x])
        elif self.model == 'lin' or self.model == 'lin-lin':
            pass

        # insert ones to first column
        self.x1 = np.concatenate((np.ones((len(self.x), 1)), self.x), axis=1)
        # transpose of x
        self.xt = np.transpose(self.x1)
        # X'X
        self.xtx = self.xt @ self.x1
        # X'X inversed
        self.xtx_inv = np.linalg.inv(self.xtx)

        # X'Y
        self.xty = self.xt @ self.y
        # (X'X)^-1(X'Y)
        self.coefficient_vector = self.xtx_inv @ self.xty
        # independent value in the model
        self.intercept = self.coefficient_vector[0]
        # estimated coefficients of the model
        self.coefficients = self.coefficient_vector[1:]
        # set the length of the series
        self.n = len(self.y)

    def y_func(self, xi):
        return float(self.intercept + sum([xij * ci for xij, ci in zip([xi], self.coefficients)]))

    def predict(self, x):
        """ Calculates predicted Y values and error values """
        # predicted Y values. Y^
        self.y_pred = np.array(
            [[self.y_func(xi[0])] for xi in x])
        # mean of Y's
        self.y_bar = float(sum(self.y) / self.n)

        # To calculate the determination coefficient r^2
        # error values
        self.errors = np.array(
            [[float(yi - yi_pred)] for yi, yi_pred in zip(self.y, self.y_pred)]
        )
        self.squared_errors = sum(np.array(
            [[e**2] for e in self.errors]
        ))
        self.total_variation = sum([(yi - self.y_bar)**2 for yi in self.y])
        self.score =  float(1 - self.squared_errors / self.total_variation)
        return self.y_pred
