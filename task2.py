# Task 2
# 
# Author: Akanksha Patel


import os
import cv2
import numpy as np
import format_data
import classifiers
import matplotlib.pyplot as plt
from svm import SVM
from knn import knn
from boosted_svm import boosted_svm
from scipy.io import loadmat
from reduce_dimension import PCA, MDA
from task1 import MLE

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

    data = data_['face']

    # View images from the first class of each dataset
    # display_images(data_, "DATA")

    dataset = 0
    n_class = 2

    # transform the data in a common format
    data = format_data.transform_data(data, dataset)

    # parameter to choose PCA or MDA
    dim_red = 0
    if dim_red == 0:
        data, data_labels = PCA(data, 0.15, 22, True)
    else:        
        data, data_labels = MDA(data, n_class, 20, 22)

    # Split data
    print("Splitting data...")
    train, train_labels, test, test_labels = format_data.split(data, data_labels, dataset, 3, 2)

    # Bayes Classifier
    mean, cov = MLE(train, train_labels, n_class, task2=True)
    bayes_acc = classifiers.bayes_classifier(test, test_labels.squeeze(), (mean, cov), n_class)
    print('Bayes: ', bayes_acc)

    # K-nn
    k = 2
    knn_ = knn(k, train, test, train_labels.squeeze(), test_labels.squeeze(), n_class)
    knn_.get_prediction()
    acc = knn_.accuracy()
    print('K-nn: ', acc)

    # SVM
    # polynomial kernel
    svm_poly = SVM(train, test, train_labels, test_labels, 'poly', 2, 1)
    print("SVM (polynomial kernel): ", svm_poly.accuracy)
    # rbf kernel
    svm_rbf = SVM(train, test, train_labels, test_labels, 'rbf', 1, 1)
    print("SVM (rbf kernel): ", svm_rbf.accuracy)
    
    # Boosted SVM
    b_svm = boosted_svm(train, train_labels, test, test_labels, 10)
    print("Boosted SVM: ", b_svm.accuracy) 



    #######################################################
    # Uncomment to generate graph for experimental values #
    #######################################################

    
    # K-nn
    # Try knn for different values to get best value of K
    # train_labels = np.reshape(train_labels, train_labels.shape[1])
    # test_labels = np.reshape(test_labels, test_labels.shape[1])
    # accuracies = []
    # indx = []
    # for i in range(1, train_labels.shape[0], 1):
    #     k = i
    #     knn1 = knn(k, train, test, train_labels, test_labels, 2)
    #     knn1.get_prediction()
    #     acc = knn1.accuracy()
    #     accuracies.append(acc)
    #     print(acc)
    #     indx.append(i)

    # plt.plot(indx, accuracies)
    # plt.xlabel('k')
    # plt.ylabel('Accuracy')
    # plt.title("k v/s accuracy")
    # plt.savefig('knn_task2.png')
    # plt.show()

    # trials for different kernel parameter (c) and marhin_parameter (s)
    # accuracies = []
    # degree = []
    # for c in range(1, 11, 1):
    #     # margins = []
    #     print(c)
    #     svm = SVM(train, test, train_labels, test_labels, 'poly', c, 1)
    #     # margins.append(s)
    #     accuracies.append(svm.accuracy)
    #     degree.append(c)
    
    # plt.plot(degree, accuracies)
        

    # plt.xlabel('Degree')
    # plt.ylabel('Accuracy')

    # plt.title("Accuracies of polynomial kernel SVM for different degree")
    # plt.savefig('svm_rbg_2222.png')
    # plt.show()


    # # Boosted SVM
    # # trials for different number of classifiers

    # n_classifiers = []
    # accuracies = []
    # for num in range(1, 20, 1):
    #     b_svm = boosted_svm(train, train_labels, test, test_labels, num)
    #     n_classifiers.append(num)
    #     accuracies.append(b_svm.accuracy)
    #     # legend.append(str(c))
        
    # plt.plot(n_classifiers, accuracies)
    # plt.xlabel('Number of classifiers')
    # plt.ylabel('Accuracy')

    # plt.title("Accuracies v/s number of classifiers boosted SVM")
    # plt.savefig('boosted_svm.png')
    # plt.show()
    




