import numpy as np
from scipy.stats import truncnorm

class Validation:
    def __init__(self, loss, mean, tau):
        self._loss = loss
        self.mean = mean
        self.query_number = 0
        self.tau = tau
    def loss(self, guess):
        self.query_number += 1
        ep = truncnorm.rvs(-1 * self.tau, self.tau)
        return(self._loss(self.mean, guess) + ep)


def mean_loss(mu, guess):
    return np.abs(guess - mu)
