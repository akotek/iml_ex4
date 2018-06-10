"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        # D is vector of distributions over the SAMPLE
        m, d = np.shape(X)
        D = np.full(shape=m, fill_value=1/m)  # init d'i...d'm as uniform dist

        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)  # ht = WL(D't, S)
            predictions = self.h[t].predict(X)  # h(x'i..x'm)
            # now we calculate eps't:
            # if y'i != h(x'i), add Weights
            eps_t = np.sum((D*(y != predictions)))
            self.w[t] = 0.5 * np.log((1 / eps_t) - 1)  # np.log==ln
            # # update Di:
            Dj_sum = np.sum((D * np.exp(-self.w[t] * y * predictions)))
            D = (D * np.exp(-self.w[t] * y * predictions)) / Dj_sum

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        # returns vector of labels y_hat
        return np.sign(np.array([
            np.sum(self.w[t] * self.h[t].predict(X)  for t in range(self.T))]))

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        return np.sum([self.predict(X) != y]) / len(y)