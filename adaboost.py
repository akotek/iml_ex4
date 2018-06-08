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
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """

        # D is vector of distributions over the SAMPLE
        m = np.size(X)
        D = np.full(shape=m, fill_value=1/m) # init uniform at first

        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)    #ht = WL(D't, S)



    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """

def read_from_txt(x_path, y_path):
    return np.genfromtxt(x_path), np.genfromtxt(y_path)

def main():
    path = "/cs/usr/kotek/PycharmProjects/iml_ex4/SynData/"

    X_train, y_train = read_from_txt(path+"X_train.txt", path+"y_train.txt")
    X_val, y_val = read_from_txt(path+"X_val.txt", path+"y_val.txt")
    X_test, y_test = read_from_txt(path+"X_test.txt", path+"y_test.txt")

    T = np.arange(0, 105, step=5)
    T = np.append(T, np.array(200))


    training_err = np.zeros(len(T))
    validation_err = np.zeros(len(T))
    test_err = np.zeros(len(T))

    # adaBoost uses a weighted trainer (WL)
    WL = ex4_tools.DecisionStump
    # TODO make in loop
    t = 0
    adaboost = AdaBoost(WL, T[t])
    adaboost.train(X_train, y_train)
    training_err[t] = adaboost.error(X_test, y_test)
    validation_err[t] = adaboost.error(X_val, y_val)
    # test_err ?



    # plt.plot(T, training_err, label="train error")
    # plt.plot(T, validation_err, label="validation error")
    # plt.legend()
    # plt.show()

   # D is a vector of m weights
    #for the samples, X\ in {R ^ {m, d}} and y is a vector of +1 and -1s of
    # size m, (similar to what we described as the weak learner in adaboost)
if __name__ == '__main__': main()