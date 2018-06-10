"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np
import ex4_tools
import matplotlib.pyplot as plt


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


def read_from_txt(x_path, y_path):
    return np.genfromtxt(x_path), np.genfromtxt(y_path)


def main():
    # path = "/cs/usr/kotek/PycharmProjects/iml_ex4/SynData/"
    path = "/cs/usr/kotek/PycharmProjects/iml_ex4/SynData/"
    X_train, y_train = read_from_txt(path + "X_train.txt", path + "y_train.txt")
    X_val, y_val = read_from_txt(path + "X_val.txt", path + "y_val.txt")
    X_test, y_test = read_from_txt(path + "X_test.txt", path + "y_test.txt")

    # -------- First part --------
    T = np.arange(5, 105, step=5)
    T = np.append(T, np.array([200]))

    training_err = np.zeros(len(T))
    validation_err = np.zeros(len(T))

    # adaBoost uses a weighted trainer (WL)
    WL = ex4_tools.DecisionStump
    for i in range(len(T)):
        adaboost = AdaBoost(WL, T[i])
        adaboost.train(X_train, y_train)
        training_err[i] = adaboost.error(X_train, y_train)
        validation_err[i] = adaboost.error(X_val, y_val)

    plt.plot(T, training_err, label="train error")
    plt.plot(T, validation_err, label="validation error")
    plt.legend()
    plt.show()
    # ------------------------

    # -------- Second part --------
    decision_T = [1, 5, 10, 100, 200]


if __name__ == '__main__': main()
