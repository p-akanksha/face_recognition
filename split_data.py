# Split data
# 
# Author: Akanksha Patel

import random
import numpy as np

random.seed(0)

# Function to split data into test and train
# Parameters:
# data: array of data
# data_type: specify the dataset (0: DATA, 1: POSE, 2: ILLUMINATION)
def split(data, data_type=0):
    # fraction of test data
    split_ratio = 0.3

    if data_type==0:
        m, n, l = data.shape
        test_num = int(split_ratio * l/3)

        # pick random samples for each class
        test_index_1 = np.array(random.sample(range(200), test_num))
        test_index_2 = np.array(random.sample(range(200), test_num))
        test_index_3 = np.array(random.sample(range(200), test_num))

        test_index_1 = 3*test_index_1
        test_index_2 = 3*test_index_2 + np.ones(test_index_1.shape)
        test_index_3 = 3*test_index_3 + 2*np.ones(test_index_1.shape)

        # test_index_1.sort()
        # test_index_2.sort()
        # test_index_3.sort()

        # select test data
        test_index = np.vstack((test_index_1, test_index_2, test_index_3)).astype('int')

        # select train data 
        train_index1 = []
        train_index2 = []
        train_index3 = []
        for i in range(600):
            if i not in test_index:
                if i%3 == 0:
                    train_index1.append(i)
                elif i%3 == 1:
                    train_index2.append(i)
                else:
                    train_index3.append(i)

        train_index1 = np.array(train_index1)
        train_index2 = np.array(train_index2)
        train_index3 = np.array(train_index3)

        train_index = np.vstack((train_index1, train_index2, train_index3))

        np.savetxt("test.csv", test_index, delimiter=",")
        np.savetxt("train.csv", train_index, delimiter=",")