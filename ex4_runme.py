"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author:
Date: May, 2018

"""

import numpy as np
import ex4_tools
import matplotlib.pyplot as plt
from adaboost import AdaBoost

def read_from_txt(x_path, y_path):
    return np.genfromtxt(x_path), np.genfromtxt(y_path)


def Q3():  # AdaBoost
    path = "/cs/usr/kotek/PycharmProjects/iml_ex4/SynData/"
    X_train, y_train = read_from_txt(path + "X_train.txt", path +
                                     "y_train.txt")
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

    # # -------- Second part --------
    decision_T = [1, 5, 10, 100, 200]


    plt.figure()
    plt.ion()
    for idx, t in enumerate(decision_T):
        adaboost = AdaBoost(WL, t)
        adaboost.train(X_train, y_train)
        plt.subplot(2, 3,  idx + 1)
        ex4_tools.decision_boundaries(adaboost, X_train, y_train, "T="+str(t))
    plt.show()
    plt.pause(5)


def Q4():  # decision trees
    # TODO - implement this function
    return


def Q5():  # spam data
    # TODO - implement this function
    return


if __name__ == '__main__':
    Q3()
    # TODO - run your code for questions 3-5
    pass
