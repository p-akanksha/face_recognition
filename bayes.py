# Bayes classifier
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import os
import cv2
import numpy as np
import split_data
from reduce_dimension import PCA, MDA




# Find Maximum Likelihood Parameters for the classes
# Parameters:
# data: dimention reduced data 
# data_type: specify the dataset (0: DATA, 1: POSE, 2: ILLUMINATION)
# def MLE(data, data_type=0):
#     if data_type == 0:





def test():
    # A = np.array([[1, 2, 3],
    #               [4, 5, 6]])
    #               # [[7, 8, 9],
    #               #  [10, 11, 12]]]);

    # print("test:")
    # B = [1, 2, 3, 4]
    # print(np.diag(B))
    # print(A.shape)
    # print(np.mean(A, axis=0))
    # print(np.mean(A, axis=1))
    # print(A)
    # print(A.shape)
    # B = np.reshape(A, (2,6))
    # print(B.shape)
    # print(B)

    C = np.array([[1, 1], 
                  [1, 1]])
    D = np.array([[2, 2], 
                  [3, 3]])

    E = 1/3*(C + D)
    print(E.shape)




x = loadmat('./Data/data.mat')
# print(len(x['face']))
# print(x['face'].shape)

y = loadmat('./Data/pose.mat')
# print(len(y['pose']))
# print(y['pose'].shape)

z = loadmat('./Data/illumination.mat')
# print(len(z['illum']))
# print(z['illum'].shape)

split_data.split(x['face'], 0)
data = x['face']
if os.path.exists("train.csv"):
    train_index = np.genfromtxt('train.csv', delimiter=',').astype('int')
    test_index = np.genfromtxt('test.csv', delimiter=',').astype('int')
else:
    split_data(data, 0)
    train_index = np.genfromtxt('train.csv', delimiter=',').astype('int')
    test_index = np.genfromtxt('test.csv', delimiter=',').astype('int')

# print(train_index)

train = np.append(data[:,:,train_index[0,0]], data[:,:,train_index[0,1]], axis=2)
# for i in range(1, train_index.shape[1]):
#     # train.append(data[:,:,train_index[0,i]])
#     # train.append(data[:,:,train_index[1,i]])
#     # train.append(data[:,:,train_index[2,i]])
#     train = np.append(train, data[:,:,train_index[0,i]], axis=1)

# train = np.array(train)
print(train.shape)

# test = data[:,:,test_index]
# train = data[:,:,train_index]


# for key in y.keys():
#   print(key)

# for i in range(13):
#     im = x['face'][:,:,i]
#     print(im.shape)

#     cv2.imshow('image',im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# test()

# PCA(train)
# MDA(x['face'])