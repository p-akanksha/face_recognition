# Support Vector Machine
# 
# Author: Akanksha Patel

# from scipy.io import loadmat
# import os
# import cv2
import numpy as np
import cvxopt
import random
from sklearn.utils import shuffle

random.seed(0)

class SVM:
    def __init__(self, train, test, train_labels, test_labels, kernel, c, s, boost=False):

        # data
        self.train_data = train
        self.train_labels = train_labels
        self.test_data = test
        self.test_labels = test_labels
        self.num_samples = train.shape[1]

        # choice of kernel and kernal parameter (c)
        self.kernel = kernel
        self.c = c

        # lagrangian multipliers
        self.a = None
        self.b = None
        self.sv_X = None
        self.sv_y = None

        # soft-margin constant
        self.s = s

        # test accuracy
        self.accuracy = None
        self.predicted_labels = None 

        # for boosted svm
        self.alpha = None
        self.boost = boost

        self.train()
        self.test()


    def poly_kernel(self, x, y):
        res = np.matmul(x.T, y)**self.c
        return res

    def rbf(self, x, y):
        res = np.exp(-np.linalg.norm(x-y, axis=0)**2/self.c**2)
        return res

    def generate_gram_matrix(self):
        K = np.zeros((self.num_samples, self.num_samples))
        if self.kernel == 'poly':
            for i in range(self.num_samples):
                for j in range(self.num_samples):
                    x = np.reshape(self.train_data[:,i], (self.train_data.shape[0], 1))
                    y = np.reshape(self.train_data[:,j], (self.train_data.shape[0], 1))
                    K[i,j] = self.poly_kernel(x, y)
        elif self.kernel == 'rbf':
            for i in range(self.num_samples):
                for j in range(self.num_samples):
                    x = np.reshape(self.train_data[:,i], (self.train_data.shape[0], 1))
                    y = np.reshape(self.train_data[:,j], (self.train_data.shape[0], 1))
                    K[i,j] = self.rbf(x, y)
        else:
            print("Wrong kernal. Choose either ploy or rbf")

        return K

    def train(self):
        K = self.generate_gram_matrix()

        # Matrices for solving dual optimization
        # max{L_D(a)} can be rewritten as
        #   min{1/2 a^T H a - 1^T a}
        #       s.t. -a_i <= 0 or s.t. a_i <= s
        #       s.t. y^t a = 0
        # where H[i, j] = y_i y_j K(x_i, x_j)

        # This form is conform to the signature of the quadratic solver provided by CVXOPT library:
        #   min{1/2 x^T P x + q^T x}
        #       s.t. G x <= h
        #       s.t. A x = b

        P = cvxopt.matrix(np.outer(self.train_labels.T, self.train_labels.T) * K)
        q = cvxopt.matrix(-np.ones(self.num_samples))
        
        # Compute G and h matrix based on the type of margin used
        if self.s:
            G = cvxopt.matrix(np.vstack((-np.eye(self.num_samples),
                                         np.eye(self.num_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(self.num_samples),
                                         np.ones(self.num_samples) * self.c)))
        else:
            G = cvxopt.matrix(-np.eye(self.num_samples))
            h = cvxopt.matrix(np.zeros(self.num_samples))
        A = cvxopt.matrix(self.train_labels.astype(np.double))
        # print(self.train_labels.shape)
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 200

        # Compute the solution using the quadratic solver
        try:
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print('Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {0:s}'.format(e))
            return
        # Extract Lagrange multipliers
        self.a = np.ravel(sol['x'])
        # print(self.a)

        self.b = 0
        for i in range(len(self.a)):
            self.b += self.train_labels[:, i]
            self.b -= np.sum(self.a * self.train_labels.T * K[i, :])
        self.b /= len(self.a)

    def predict(self, x, i):
        s = 0

        if self.kernel == 'poly':
            for j in range(self.num_samples):
                x_i = np.reshape(self.train_data[:, j], (self.train_data.shape[0], 1))
                s = s + self.a[i]*self.train_labels[:,i]*self.poly_kernel(x, x_i)

        elif self.kernel == 'rbf':
            for j in range(self.num_samples):
                x_i = np.reshape(self.train_data[:, j], (self.train_data.shape[0], 1))
                s = s + self.a[i]*self.train_labels[:,i]*self.rbf(x, x_i)
        else:
            print("What did you choose??")

        return np.sign(s)

    def test(self):
        if not self.boost:

            predicted_labels = []

            correct = 0
            for i in range(self.test_data.shape[1]):
                x = np.reshape(self.test_data[:, i], (self.test_data.shape[0],1))
                pred_label = self.predict(x, i)

                predicted_labels.append(pred_label)

                if pred_label == self.test_labels[:, i]:
                    correct = correct + 1

            self.predicted_labels = np.array(predicted_labels)        

            self.accuracy = correct/float(self.test_data.shape[1])

        else:
            predicted_labels = []

            correct = 0
            for i in range(self.train_data.shape[1]):
                x = np.reshape(self.train_data[:, i], (self.train_data.shape[0],1))
                pred_label = self.predict(x, i)

                predicted_labels.append(pred_label)

                if pred_label == self.train_labels[:, i]:
                    correct = correct + 1

            self.predicted_labels = np.array(predicted_labels)        

            self.accuracy = correct/float(self.train_data.shape[1])
