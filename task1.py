# Task 1: Subject Identification
# 
# Author: Akanksha Patel


import os
import cv2
import numpy as np
import classifiers
import format_data
import matplotlib.pyplot as plt
from knn import knn
from svm import SVM
from boosted_svm import boosted_svm
from scipy.io import loadmat
from reduce_dimension import PCA, MDA
from sklearn.covariance import LedoitWolf

# Find Maximum Likelihood Parameters for the classes
# Parameters:
# data: dimention reduced data 
# data_type: specify the dataset (0: DATA, 1: POSE, 2: ILLUMINATION)
def MLE(train, train_labels, n_class, data_type=0):
    # print(train.shape)
    m, l = train.shape

    data_class = []

    for i in range(n_class):
        cl = []
        for j in range(train_labels.shape[0]):
            if train_labels[j] == i:
                cl.append(train[:,j])
        data_class.append(np.asarray(cl))

    data_class = np.asarray(data_class)
    data_class_2 = np.zeros((m, l/n_class, n_class))
    for i in range(l/n_class):
        for j in range(n_class):
            data_class_2[:,i,j] = data_class[j,i,:]

    # compute mean of classes 
    mean = np.zeros((m, 1, n_class))
    for c in range(n_class):
        d = data_class_2[:,:,c]
        mean[:, :, c] = np.reshape(np.mean(d, axis=1), (d.shape[0], 1))

    # covariance of different features
    cov = np.zeros((m, m, n_class))
    for c in range(n_class):
        d = data_class_2[:,:,c].T
        sig = LedoitWolf().fit(d).covariance_
        X = np.random.multivariate_normal(mean=mean[:,:,c].squeeze(),
                                              cov=sig,
                                              size=50)
        cov[:,:,c] = LedoitWolf().fit(d).covariance_
        # if np.linalg.det(cov[:,:,c]) == 0:
        #     print("00000000000")
        # else:
        #     print("dhsjhj")

    # print(mean.shape)

    return mean, cov


def get_PCA_test(data, pca):
    m, l = data.shape

    flattened_images = []
    # To-do: change with reshape
    for i in range(l):
        im = data[:,i]
        flattened_images.append(im)

    flattened_images = np.array(flattened_images).T

    # print("bayes: ", basis.shape)
    # print("bayes: ", flattened_images.shape)

    proj_data = pca.transform(flattened_images.T)
    proj_data = proj_data.T

    return proj_data


def generate_data_labels(data, n_class):
    m, l = data.shape

    labels = np.zeros((1, l))

    for i in range(l):
        labels[0][i] = i%n_class

    return labels

# view images of the first class
def display_images(data, type):
    if type == 'DATA':
        for i in range(3):
            im = data[:,:,i]

            cv2.imshow('image',im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif type == 'POSE':
        for i in range(13):
            # cv2 expects images in uint8 format
            im = data[:,:,i,0].astype(np.uint8)

            cv2.imshow('image',im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif type == 'ILLUMINATION':
        for i in range(21):
            # cv2 expects images in uint8 format
            im = data[:,i,0].astype(np.uint8)
            im = np.reshape(im, (40, 48)).T

            cv2.imshow('image',im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Wrong data_type selected. Select one from: DATA, POSE, ILLUMINATION")


if __name__ == '__main__':

    # Load data
    data_ = loadmat('./Data/data.mat')
    pose_ = loadmat('./Data/pose.mat')
    illum_ = loadmat('./Data/illumination.mat')

    data_ = data_['face']
    pose_ = pose_['pose']
    illum_ = illum_['illum']

    # View images from the first class of each dataset
    # display_images(data_, "DATA")
    # display_images(pose_, "POSE")
    # display_images(illum_, "ILLUMINATION")

    # Choose dataset
    # 0 : DATA
    # 1 : POSE
    # 2 : ILLUMINATION
    dataset = 1

    if dataset == 0:
        data = data_
        n_class = 200
    elif dataset == 1:
        data = pose_
        n_class = 68
    elif dataset == 2:
        data = illum_
        n_class = 68

    # transform the data in a common format
    data = format_data.transform_data(data, dataset)

    # parameter to choose PCA or MDA
    dim_red = 0
    if dim_red == 0:
        data, data_labels = PCA(data, 0.15)
    else:        
        data, data_labels = MDA(data, n_class, 20)

    print("Splitting data...")
    train, train_labels, test, test_labels = format_data.split(data, data_labels, dataset, n_class, 1)

    # knn classifier
    k = 1
    knn_ = knn(k, train, test, train_labels, test_labels, n_class)
    knn_.get_prediction()
    acc = knn_.accuracy()
    print(acc)

    # Bayes classifier
    mean, cov = MLE(train, train_labels, n_class)
    classifiers.bayes_classifier(test, test_labels, (mean, cov), n_class)


    # lw = 5.0
    # legend = []
    # for thresh in range(35, 5,-10):

    #     # reduce dimensions
    #     data_pca = PCA(data, thresh/float(100))

    #     # split data in to test and train 
    #     # if os.path.exists("train_" + str(dataset) + ".csv"):
    #     #     print("Loading saved data...")
    #     #     train = np.genfromtxt("train_" + str(dataset) + ".csv", delimiter=',').astype('int')
    #     #     test = np.genfromtxt("test_" + str(dataset) + ".csv", delimiter=',').astype('int')
    #     #     train_labels = np.genfromtxt("train_labels_" + str(dataset) + ".csv", delimiter=',').astype('int')
    #     #     test_labels = np.genfromtxt("test_labels_" + str(dataset) + ".csv", delimiter=',').astype('int')
    #     # else:
    #     print("Splitting data...")
    #     train, train_labels, test, test_labels = format_data.split(data, dataset, 1)

    #     # print(train.shape)
    #     # print(test.shape)
    #     # print(train_labels.shape)
    #     # print(test_labels.shape)

    #     # Reduce dimensions 
    #     # train, basis, pca = PCA(train)

    #     # test = get_PCA_test(test, pca)

    #     # Try knn for different values to get best value of K
    #     accuracies = []
    #     indx = []
    #     for i in range(1, 250, 1):
    #         k = i
    #         knn1 = knn(k, train, test, train_labels, test_labels, n_class)
    #         knn1.get_prediction()
    #         acc = knn1.accuracy()
    #         accuracies.append(acc)
    #         indx.append(i)

    #     plt.plot(indx, accuracies)
    #     plt.title("threshold " + str(thresh))
    #     plt.savefig('knn_' + str(dataset) + '_' + str(thresh) + '.png')
    #     np.savetxt("accuracies_" + str(dataset) + '_' + str(thresh) + ".csv", accuracies, delimiter=",")
    #     plt.show()

    # # plt.xlabel('k')
    # # plt.ylabel('Accuracy')
    

    # accuracies = []
    # indx = []
    # for i in range(1, 50, 1):
    #     k = i
    #     knn1 = knn(k, train, test, train_labels, test_labels, n_class)
    #     knn1.get_prediction()
    #     acc = knn1.accuracy()
    #     accuracies.append(acc)
    #     print(acc)
    #     indx.append(i)

    # plt.plot(indx, accuracies)
    # plt.title("Accuracy v/s K")
    # plt.savefig('knn_' + str(dataset) + '.png')
    # # np.savetxt("accuracies_" + str(dataset) + '_' + str(thresh) + ".csv", accuracies, delimiter=",")
    # plt.show()

    # knn classifier
    # k = 1
    # knn_ = knn(k, train, test, train_labels, test_labels, n_class)
    # knn_.get_prediction()
    # acc = knn_.accuracy()
    # print(acc)


    # train = np.dstack((data[:,:,train_index[0,0]], data[:,:,train_index[1,0]], data[:,:,train_index[2,0]]))
    # for i in range(1, train_index.shape[1]):
    #     train = np.dstack((train, data[:,:,train_index[0,i]], data[:,:,train_index[1,i]], data[:,:,train_index[2,i]]))

    # test = np.dstack((data[:,:,test_index[0,0]], data[:,:,test_index[1,0]], data[:,:,test_index[2,0]]))
    # for i in range(1, test_index.shape[1]):
    #     test = np.dstack((test, data[:,:,test_index[0,i]], data[:,:,test_index[1,i]], data[:,:,test_index[2,i]]))





    # print(train.shape)
    # mean, cov = MLE(train, train_labels, n_class)

    # classifiers.bayes_classifier(test, test_labels, (mean, cov), n_class)






