# Task 1: Subject Identification
# 
# Author: Akanksha Patel


# To-do: Is Data Reduction applied on the whole data befor the split?

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
def MLE(train, train_labels, n_class, data_type=0, task2=False):
    # To-do: Do something about the singular covariance matrix
    m, l = train.shape
    train_labels = train_labels.squeeze()

    data_class = []

    if task2:
        class1 = []
        class2 = []

        for i in range(n_class):
            for j in range(train_labels.shape[0]):
                if train_labels[j] == 1:
                    class1.append(train[:,j])
                else:
                    class2.append(train[:,j])
        
        class1 = np.array(class1).T
        class2 = np.array(class2).T

        # compute mean of classes 
        mean = np.zeros((m, 1, n_class))
        mean[:,:,0] = np.reshape(np.mean(class1, axis=1), (class1.shape[0], 1))
        mean[:,:,1] = np.reshape(np.mean(class2, axis=1), (class2.shape[0], 1))

        # covariance of different features
        cov = np.zeros((m, m, n_class))
        sig1 = LedoitWolf().fit(class1.T).covariance_
        X1 = np.random.multivariate_normal(mean=mean[:,:,0].squeeze(),
                                              cov=sig1,
                                              size=50)
        cov[:,:,0] = LedoitWolf().fit(X1).covariance_

        sig2 = LedoitWolf().fit(class2.T).covariance_
        X2 = np.random.multivariate_normal(mean=mean[:,:,1].squeeze(),
                                              cov=sig2,
                                              size=50)
        cov[:,:,1] = LedoitWolf().fit(X2).covariance_

        return mean, cov

    for i in range(n_class):
        cl = []
        for j in range(train_labels.shape[0]):
            if train_labels[j] == i:
                cl.append(train[:,j])
        data_class.append(np.asarray(cl))

    data_class = np.asarray(data_class)
    data_class_2 = np.zeros((m, int(l/n_class), n_class))
    for i in range(int(l/n_class)):
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
        cov[:,:,c] = LedoitWolf().fit(X).covariance_

    return mean, cov

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
    dataset = 2

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
    dim_red = 1
    if dim_red == 0:
        if os.path.exists("PCA_" + str(dataset) + ".csv") and os.path.exists("PCA_labels_" + str(dataset) + ".csv"):
            print("Loading saved PCA...")
            data = np.genfromtxt("PCA_" + str(dataset) + ".csv", delimiter=',').astype('float')
            data_labels = np.genfromtxt("PCA_labels_" + str(dataset) + ".csv", delimiter=',').astype('float')
        else:
            print("Applying PCA...")
            data, data_labels = PCA(data, 0.35, dataset)
    else:
        if os.path.exists("MDA_" + str(dataset) + ".csv") and os.path.exists("MDA_labels_" + str(dataset) + ".csv"):
            print("Loading saved MDA...")
            data = np.genfromtxt("MDA_" + str(dataset) + ".csv", delimiter=',').astype('float')
            data_labels = np.genfromtxt("MDA_labels_" + str(dataset) + ".csv", delimiter=',').astype('float')
        else:
            print("Applying MDA...")    
            data, data_labels = MDA(data, n_class, 20, dataset)

    print("Splitting data...")
    train, train_labels, test, test_labels = format_data.split(data, data_labels, dataset, n_class, 1)

    # knn classifier
    k = 1
    knn_ = knn(k, train, test, train_labels, test_labels, n_class)
    knn_.get_prediction()
    knn_acc = knn_.accuracy()
    print("knn Accuracy: ", knn_acc)

    # Bayes classifier
    mean, cov = MLE(train, train_labels, n_class)
    bayes_acc = classifiers.bayes_classifier(test, test_labels, (mean, cov), n_class)
    print("Bayes Accuracy: ", bayes_acc)


    ################################################
    # Uncomment for generating experimental graphs #
    ################################################

    '''
    # knn with PCA
    legend = []
    for thresh in range(35, 5,-10):

        # reduce dimensions
        data, data_labels = PCA(data, thresh/float(100), dataset)
        
        # Split data in training and testing
        print("Splitting data...")
        train, train_labels, test, test_labels = format_data.split(data, data_labels, dataset, n_class, 1)

        # Try knn for different values to get best value of K
        accuracies = []
        indx = []
        for i in range(1, 250, 1):
            k = i
            knn1 = knn(k, train, test, train_labels, test_labels, n_class)
            knn1.get_prediction()
            acc = knn1.accuracy()
            accuracies.append(acc)
            indx.append(i)

        plt.plot(indx, accuracies)
        plt.title("threshold " + str(thresh))
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.savefig('knn_' + str(dataset) + '_' + str(thresh) + '.png')
        plt.show()
    '''

    # plt.xlabel('k')
    # plt.ylabel('Accuracy')
    

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

    # split data in to test and train 
        # if os.path.exists("train_" + str(dataset) + ".csv"):
        #     print("Loading saved data...")
        #     train = np.genfromtxt("train_" + str(dataset) + ".csv", delimiter=',').astype('int')
        #     test = np.genfromtxt("test_" + str(dataset) + ".csv", delimiter=',').astype('int')
        #     train_labels = np.genfromtxt("train_labels_" + str(dataset) + ".csv", delimiter=',').astype('int')
        #     test_labels = np.genfromtxt("test_labels_" + str(dataset) + ".csv", delimiter=',').astype('int')
        # else:






