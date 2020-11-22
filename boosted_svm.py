# Boosted SVM
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import os
import cv2
import numpy as np
from svm import SVM
from tqdm import tqdm, trange


class boosted_svm():

    def __init__(self, train_data, train_labels, test_data, test_labels, iterations = 10,
                 max_epochs = 10000, learning_rate = 0.01, threshold = 0.5):
        # data
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.n_samples = train_data.shape[1]

        # constants for training
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.iterations = iterations

        # weights
        self.weights = np.full(self.n_samples, 1/self.n_samples)
        self.classifiers = []

        self.train()
        self.test()

    def train(self):
        weights = np.full(self.n_samples, 1/self.n_samples)

        for i in trange(self.iterations):

            count = 0
            error = np.inf

            while error > self.threshold:
                svm = SVM(self.train_data, self.test_data, self.train_labels, self.test_labels, 'rbf', 1, 1, True)

                error = 0
                # print(self.n_samples)
                for i in range(self.n_samples):
                    if self.train_labels[:, i] != svm.predicted_labels[i]:
                        error = error + weights[i]  

                if error > self.threshold:
                    print(error)
                    print("continuing...")
                    continue

                svm.alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))

                weights = np.exp(-1.0 * svm.alpha * self.train_labels.squeeze() * svm.predicted_labels.squeeze())
                weights = weights / np.sum(weights)

                self.classifiers.append(svm)

    def test(self):
        predicted_labels = []

        correct = 0
        for i in range(self.test_data.shape[1]):
            x = np.reshape(self.test_data[:, i], (self.test_data.shape[0],1))
            
            label = 0
            for cl in self.classifiers:
                cl_label = cl.predict(x, i)
                label = label + cl.alpha*cl_label

            predicted_labels.append(np.sign(label))

            if np.sign(label) == self.test_labels[:, i]:
                correct = correct + 1

            else:
                print("incorrect: ", np.sign(label) , self.test_labels[:, i])

        self.accuracy = correct/float(self.test_data.shape[1])



                
                

