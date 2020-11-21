# Classifiers
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import os
import cv2
import numpy as np
from scipy.stats import multivariate_normal


def get_prob(mean, cov, x):
    sig = np.linalg.det(cov)
    l = cov.shape[0]
    if sig == 0:
        print("Determinent zero")
        return 0

    N = np.sqrt((2*np.pi)**l*sig)
    ep = np.matmul((x-mean).T, cov)
    ep1 = np.matmul(ep, (x-mean))

    prob = 1/N * np.exp(ep1)

    return prob

def bayes_classifier(data, labels, params, n_class, data_type=0):
    m, l = data.shape

    # flattened_images = []
    # # To-do: change with reshape
    # for i in range(l):
    #     im = np.ravel(data[:,:,i])
    #     flattened_images.append(im)

    # flattened_images = np.array(flattened_images).T

    # print("bayes: ", basis.shape)
    # print("bayes: ", flattened_images.shape)

    # proj_data = pca.transform(flattened_images.T)
    # proj_data = proj_data.T
    # proj_data = np.matmul(basis.T, flattened_images)

    # print("bayes:", proj_data.shape)
    mean, cov = params
    # print(mean.shape)

    correct = 0
    for i in range(data.shape[1]):
        x = np.reshape(np.array(data[:, i]), (len(data[:, i]), 1))
        prob = np.zeros(n_class)
        for j in range(n_class):
            prob[j] = multivariate_normal.pdf(x.squeeze(), mean[:,:,j].squeeze(), cov[:,:,j])

        c = np.argmax(prob)

        if(labels[i] == c):
            correct = correct + 1

    print("Accuracy, ", float(correct)/l)

    return float(correct)/l
