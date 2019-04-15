import numpy as np
from scipy.stats import truncnorm

class Validation:
    def __init__(self, loss, mean, tau):
        self._loss = loss
        self.mean = mean
        self.query_number = 0
        self.tau = tau
    def trunc_gauss(self):
        if self.tau == 0:
            return 0
        else:
            return truncnorm.rvs(-1 * self.tau, self.tau)
    def loss(self, guess):
        self.query_number += 1
        return(self._loss(self.mean, guess) + self.trunc_gauss())


def mean_loss(mu, guess):
    return np.linalg.norm(guess - mu)
