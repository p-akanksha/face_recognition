# K nearest neighbour classifier
# 
# Author: Akanksha Patel

import numpy as np

class knn:
    def __init__(self, k, train, test, n):
        self.k = k
        self.num_labels = n
        self.train_data = train
        self.test_data = test
        self.predictions = None

    def get_distance(self, x):
        rep_vec = np.tile(x, self.train_data.shape[1])
        diff = self.train_data - rep_vec
        dists = np.linalg.norm(diff, axis=0)

        return dists
    
    # Function to calculate prediction for a single point
    def get_predicition_util(self, x):
        # calculate distance from all the points in train data
        dists = self.get_distance(x)

        # get closest k points
        indices = np.argsort(dists)
        indices = indices[:self.k]

        # get the class with maximum labels
        label_count = np.zeros(self.num_labels)
        for j in range(self.k):
            c = indices[j]%self.num_labels
            label_count[c] = label_count[c] + 1

        pred = np.argmax(label_count)

        return pred


    # Function to calculate predictions for test_data
    def get_prediction(self):
        predictions = []

        for i in range(self.test_data.shape[1]):
            x = np.reshape(self.test_data[:,i], (self.test_data.shape[0], 1))
            pred = self.get_predicition_util(x)
            predictions.append(pred)

        predictions = np.array(predictions)
        self.predictions = predictions

    # Function to calculate accuracy of the predictions
    def accuracy(self):
        l = len(self.predictions)
        correct = 0

        for i in range(l):
            if (self.predictions[i] == i%self.num_labels):
                correct = correct + 1

        return correct/float(l)