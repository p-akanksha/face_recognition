# Classifiers
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import os
import cv2
import numpy as np

def get_prob(mean, cov, x):
    sig = np.linalg.det(cov)
    l = cov.shape[0]
    if sig == 0:
        print("What the hell??")
        return 0

    N = np.sqrt((2*np.pi)**l*sig)
    ep = np.matmul((x-mean).T, cov)
    ep1 = np.matmul(ep, (x-mean))

    prob = 1/N * np.exp(ep1)

    return prob


def bayes_classifier(data, params, basis, pca, data_type=0):
    m, n, l = data.shape

    flattened_images = []
    # To-do: change with reshape
    for i in range(l):
        im = np.ravel(data[:,:,i])
        flattened_images.append(im)

    flattened_images = np.array(flattened_images).T

    # print("bayes: ", basis.shape)
    # print("bayes: ", flattened_images.shape)

    proj_data = pca.transform(flattened_images.T)
    proj_data = proj_data.T
    # proj_data = np.matmul(basis.T, flattened_images)

    print("bayes:", proj_data.shape)
    mean, cov = params
    # print(mean.shape)

    correct = 0
    for i in range(proj_data.shape[1]):
        x = np.reshape(np.array(proj_data[:, i]), (len(proj_data[:, i]), 1))
        prob1 = get_prob(mean[:,:,0], cov[:,:,0], x)
        prob2 = get_prob(mean[:,:,1], cov[:,:,1], x)
        prob3 = get_prob(mean[:,:,2], cov[:,:,2], x)

        # print(prob1)

        prob = np.asarray([prob1, prob2, prob3])
        # print(prob)

        c = np.argmax(prob)

        # print(c, i%3)
        if(i%3 == c):
            correct = correct + 1

    print(correct)
