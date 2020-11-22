# Functions to reduce data dimensions
# 
# Author: Akanksha Patel

import cv2
import numpy as np
from sklearn import decomposition
from sklearn.covariance import LedoitWolf

def PCA(images, thresh, dataset, task2 = False):
    m, n, l = images.shape

    data = []
    data_labels = []

    if task2: 
        for i in range(l):
            for j in range(n):
                data.append(images[:,j,i])
                data_labels.append(j)
    else:
        for i in range(l):
            for j in range(n):
                data.append(images[:,j,i])
                data_labels.append(i)

    data = np.asarray(data).T
    data_labels = np.asarray(data_labels).T

    mean = np.mean(data, axis=1)
    mean = np.reshape(mean, (len(mean), 1))
    zero_cetered_x = data - np.tile(mean, l*n)

    u, s, vh = np.linalg.svd(zero_cetered_x, full_matrices=True)

    sm = np.sum(s) 
    
    ss = 0
    for i in range(s.shape[0]):
        ss = ss + s[i]
        if ss/sm > thresh:
            num = i
            break

    print("Using " + str(num) + " features out of " + str(m))

    sigma = np.zeros((num, vh.shape[0]))
    for i in range(min(vh.shape[0],num)):
        sigma[i][i] = s[i]
    x_new = np.matmul(sigma, vh)

    np.savetxt("PCA_" + str(dataset) + ".csv", x_new, delimiter=",")
    np.savetxt("PCA_labels_" + str(dataset) + ".csv", data_labels, delimiter=",")

    return x_new, data_labels

    
def MDA(data, n_class, num, dataset):
    m, n, l = data.shape

    # compute mean of classes 
    mean = np.zeros((m, 1, n_class))
    for c in range(n_class):
        d = data[:,:,c]
        mean[:, :, c] = np.reshape(np.mean(d, axis=1), (d.shape[0], 1))

    # covariance of different features
    cov = np.zeros((m, m, n_class))
    for c in range(n_class):
        d = data[:,:,c].T
        sig = LedoitWolf().fit(d).covariance_
        X = np.random.multivariate_normal(mean=mean[:,:,c].squeeze(),
                                              cov=sig,
                                              size=50)
        cov[:,:,c] = LedoitWolf().fit(d).covariance_

    # within class scatter matrix
    within_class_sm = np.zeros_like(cov[:,:,0])
    for c in range(n_class):
        within_class_sm = within_class_sm + cov[:,:,c]

    within_class_sm = 1/float(n_class) * within_class_sm

    # overall mean
    overall_mean = np.zeros_like(mean[:,:,0])
    for c in range(n_class):
        overall_mean = overall_mean + mean[:,:,c]

    overall_mean = 1/float(n_class) * overall_mean

    # between class scatter matrix
    between_class_sm = np.zeros_like(np.matmul(mean[:,:,0],mean[:,:,0].T))
    for c in range(n_class):
        between_class_sm = between_class_sm + np.matmul(mean[:,:,c],mean[:,:,c].T)

    between_class_sm = 1/float(n_class) * between_class_sm

    if(np.linalg.det(within_class_sm) != 0):
        within_class_inv = np.linalg.inv(within_class_sm)
        J = np.matmul(np.linalg.inv(within_class_sm), between_class_sm)
    else:
        print("Determinent = 0 !!!!!!!!!")

    eig_value, eig_vector = np.linalg.eig(J)

    # Sorting eigen value and corresponding eig vector in descending order
    eig_value = eig_value.real
    eig_value = eig_value / np.sum(eig_value)
    index = np.argsort(eig_value)
    index = index[::-1]
    eig_value_sort = eig_value[index]
    eig_vector_sort = eig_vector[:, index]

    # Taking first num components 
    eig_vector_sort = eig_vector_sort.real[:,:num]

    X = np.zeros((m, n*l))
    X_labels = np.zeros((n*l, 1))
    for i in range(l):
        for j in range(n):
            X[:,n*i+j] = data[:,j,i]
            X_labels[n*i+j,:] = i

    X_new = np.matmul(X.T, eig_vector_sort).T

    np.savetxt("MDA_" + str(dataset) + ".csv", X_new, delimiter=",")
    np.savetxt("MDA_labels_" + str(dataset) + ".csv", X_labels, delimiter=",")

    return X_new, X_labels 