# Classifiers
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import os
import cv2
import numpy as np
from scipy.stats import multivariate_normal

def bayes_classifier(data, labels, params, n_class, data_type=0):
    m, l = data.shape
    mean, cov = params

    correct = 0
    for i in range(data.shape[1]):
        x = np.reshape(np.array(data[:, i]), (len(data[:, i]), 1))
        prob = np.zeros(n_class)
        for j in range(n_class):
            prob[j] = multivariate_normal.pdf(x.squeeze(), mean[:,:,j].squeeze(), cov[:,:,j])

        c = np.argmax(prob)

        if(labels[i] == c):
            correct = correct + 1
        else:
            print("xxxxxxxxxxxxx")

    return float(correct)/l
