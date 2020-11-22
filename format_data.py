# Split data
# 
# Author: Akanksha Patel

import random
import cv2
import numpy as np

random.seed(0)

# Format the dataset in a common format
# Returns a m*n matrix with n flattened images as column vectors

def transform_data(data, dataset):
    if dataset == 0:
        m, n, l = data.shape
        result = np.zeros((m*n, 3, int(l/3)))

        for i in range(int(l/3)):
            result[:, 0, i] = data[:, :, 3*i].flatten()
            result[:, 1, i] = data[:, :, 3*i+1].flatten()
            result[:, 2, i] = data[:, :, 3*i+2].flatten()

        return result


    # pose(:,:,i,j) gives i^th image of j^th subject.
    elif dataset == 1:
        p, q, r, l = data.shape
        result = np.zeros((p*q, r, l))

        for j in range(l):
            for i in range(r):
                result[:,i,j] = data[:,:,i,j].flatten()

        return result


    elif dataset == 2:
        return data

    else: 
        print("Invalid dataset selected. Select one from: 0, 1, or 2.")

    return None

# split data into train and test
# split_type:
# 0: train = (neutral, expression), test=(illumination)
# 1: train = (neutral, illumination), test=(expression) 
# 2: train = (expression, illumination), test=(neutral)
# 3: random assignment
def split_data(data, split_type = 0):
    m, n, l = data.shape

    if split_type == 0:
        train = np.hstack((data[:,0,:], data[:,1,:]))
        test = np.reshape(data[:,2,:], (m,l))

        train_labels = np.hstack((range(200), range(200)))
        test_labels = np.asarray(range(200))

        # print(train.shape)
        # print(test.shape)
        # print(train_labels)
        # print(test_labels)

        return train, train_labels, test, test_labels

    elif split_type == 1:
        train = np.hstack((data[:,0,:], data[:,2,:]))
        test = np.reshape(data[:,1,:], (m,l))

        train_labels = np.hstack((range(200), range(200)))
        test_labels = np.asarray(range(200))

        # print(train.shape)
        # print(test.shape)
        # print(train_labels)
        # print(test_labels)

        return train, train_labels, test, test_labels

    elif split_type == 2:
        train = np.hstack((data[:,1,:], data[:,2,:]))
        test = np.reshape(data[:,0,:], (m,l))

        train_labels = np.hstack((range(200), range(200)))
        test_labels = np.asarray(range(200))

        # print(train.shape)
        # print(test.shape)
        # print(train_labels.shape)
        # print(test_labels.shape)

        return train, train_labels, test, test_labels

    elif split_type == 3:
        train = np.zeros((m, 2*l))
        train_labels = np.zeros(2*l)
        test = np.zeros((m, l))
        test_labels = np.zeros(l)

        for i in range(l):
            train_idx = np.array(random.sample(range(3), 2))
            train[:,2*i] = data[:,train_idx[0],i]
            train[:,2*i+1] = data[:,train_idx[1],i]
            train_labels[2*i] = i
            train_labels[2*i + 1] = i

            for j in range(3):
                if j not in train_idx:
                    test[:,i] = data[:,j,i]
                    test_labels[i] = i

        # print(train.shape)
        # print(test.shape)
        # print(train_labels)
        # print(test_labels)

        return train, train_labels, test, test_labels

    else: 
        print("Invalid split_type.")

# Helper function to split data in test and train 
def split_util(data, data_type):
    m, n, l = data.shape
    if data_type == 1:
        # split ratio = 9:4
        num_train = 9
        num_test = 4
        
    elif data_type == 2:
        # split ratio = 15:6
        num_train = 15
        num_test = 6

    else:
        print("Incorrect data type ", data_type)
    
    train = np.zeros((m, num_train*l))
    train_labels = np.zeros(num_train*l)
    test = np.zeros((m, num_test*l))
    test_labels = np.zeros(num_test*l)

    for i in range(l):
        train_idx = np.array(random.sample(range(n), num_train))

        for k in range(num_train):
            train[:,num_train*i+k] = data[:,train_idx[k],i]
            train_labels[num_train*i+k] = i

        count = 0
        for j in range(n):
            if j not in train_idx:
                test[:,num_test*i+count] = data[:,j,i]
                test_labels[num_test*i+count] = i
                count = count + 1

    return train, train_labels, test, test_labels

# Function to split data into test and train
# Parameters:
# data: array of data
# data_type: specify the dataset (0: DATA, 1: POSE, 2: ILLUMINATION)
def split(data, data_labels, data_type, n_class, task):
    m, l = data.shape

    data_class = []

    for i in range(n_class):
        cl = []
        for j in range(l):
            if data_labels[j] == i:
                cl.append(data[:,j])
        data_class.append(np.asarray(cl))

    data_class = np.asarray(data_class)

    data_class_2 = np.zeros((m, int(l/n_class), n_class))
    for i in range(int(l/n_class)):
        for j in range(n_class):
            data_class_2[:,i,j] = data_class[j,i,:]

    data = data_class_2

    if task == 1:
        if data_type == 0:
            train, train_labels, test, test_labels = split_data(data, 1)

        elif data_type == 1:
            train, train_labels, test, test_labels = split_util(data, data_type)

        elif data_type == 2:
            train, train_labels, test, test_labels = split_util(data, data_type)
        else: 
            print("Invalid data_type.")
            return None, None, None, None

        np.savetxt("train_" + str(data_type) +".csv", train, delimiter=",")
        np.savetxt("test_" + str(data_type) +".csv", test, delimiter=",")
        np.savetxt("train_labels_" + str(data_type) +".csv", train_labels, delimiter=",")
        np.savetxt("test_labels_" + str(data_type) +".csv", test_labels, delimiter=",")

        return train, train_labels, test, test_labels

    elif task == 2:
        # fraction of test data
        split_ratio = 0.3
        
        if data_type != 0:
            print("Error! Invalid dataset")

        # m - num of features
        # n - num of samples in each class
        # l - num of class
        m, n, l = data.shape
        test_num = int(split_ratio * n)
        train_num = n - test_num

        # pick random samples for each class
        test_index = np.asarray([np.array(random.sample(range(n), test_num)), 
                                 np.array(random.sample(range(n), test_num)), 
                                 np.array(random.sample(range(n), test_num))])

        test = np.zeros((m, 3*test_num))
        test_labels = np.zeros((1, 3*test_num))

        # assign -1 label to facial expression
        for i in range(3):
            for j in range(test_num):
                test[:, test_num*i+j] = data[:, test_index[i, j], i]
                if i != 1:
                    test_labels[:, test_num*i+j] = -1
                else:
                    test_labels[:, test_num*i+j] = 1                    


        # select train data 
        train_index = np.zeros((3, train_num))
        train = np.zeros((m, 3*train_num))
        train_labels = np.zeros((1, 3*train_num))
        for i in range(3):
            count = 0
            for j in range(n):
                if j not in test_index[i,:]:
                    train[:, train_num*i+count] = data[:, j, i]
                    if i != 1:
                        train_labels[:, test_num*i+count] = -1
                    else:
                        train_labels[:, test_num*i+count] = 1
                    count = count+1

        np.savetxt("train_task2.csv", train, delimiter=",")
        np.savetxt("test_task2.csv", test, delimiter=",")
        np.savetxt("train_labels_task2.csv", train_labels, delimiter=",")
        np.savetxt("test_labels_task2.csv", test_labels, delimiter=",")

        return train, train_labels, test, test_labels

    else:
        print("Invalid task selection.")