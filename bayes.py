# Bayes classifier
# 
# Author: Akanksha Patel

from scipy.io import loadmat
import cv2
import random
import numpy as np

def split_data(data, type=0):
    test = []
    train = []
    split_ratio = 0.3
    if type==0:
        m, n, l = data.shape
        test_num = int(split_ratio * l/3)

        test_index_1 = np.array(random.sample(range(200), test_num))
        test_index_2 = np.array(random.sample(range(200), test_num))
        test_index_3 = np.array(random.sample(range(200), test_num))

        test_index_1 = 3*test_index_1
        test_index_2 = 3*test_index_2 + np.ones(test_index_1.shape)
        test_index_3 = 3*test_index_3 + 2*np.ones(test_index_1.shape)

        # test_index = 

        test = data[:,:,test_index_1]



def PCA(images):
    m, n, l = images.shape
    thresh = 0.05 # 0.01

    print(images.shape)

    flattened_images = []
    # To-do: change with reshape
    for i in range(l):
        im = np.ravel(images[:,:,i])
        flattened_images.append(im)

    flattened_images = np.array(flattened_images).T

    # Make data zero-centric
    mean = np.mean(flattened_images, axis=1)
    mean = np.reshape(mean, (len(mean),1))
    zere_cetered_x = flattened_images - np.tile(mean, l)

    u, s, vh = np.linalg.svd(zere_cetered_x, full_matrices=True)

    sigma = np.zeros((u.shape[0],vh.shape[0]))
    for i in range(min(u.shape[0],vh.shape[0])):
        sigma[i][i] = s[i]
    x_new = np.matmul(sigma, vh)

    for i in range(x_new.shape[1]):
        img_flat = x_new[:,i]

        # For vizualization
        img_min = np.min(img_flat)
        img_max = np.max(img_flat)
        # print("Min, max: ", img_min, img_max)
        xx = img_max - img_min
        norm_factor = 1/xx
        img_min_arr = img_min * np.ones(img_flat.shape)
        img_flat = norm_factor * (img_flat - img_min_arr)
        print(img_flat.shape)

        img = np.reshape(img_flat, (m, n))

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    # sm = np.sum(s) 

    # print(sm)
    # sh = s.shape
    # for i in range(sh[0]):
    #     # print((s[i]/sm)*100)
    #     if (s[i]/sm)*100 < thresh:
    #         print(i)
    #         break

def MDA(images, data = 0):
    m, n, l = images.shape

    if data == 0: 
        exp = []
        neutral = []
        ill = []

        for i in range(0, l, 3):
            exp.append(np.ravel(images[:,:,i]))
            neutral.append(np.ravel(images[:,:,i+1]))
            ill.append(np.ravel(images[:,:,i+2]))

        # convert to np array
        exp = np.array(exp).T
        neutral = np.array(neutral).T
        ill = np.array(ill).T

        print(exp.shape)
        print(neutral.shape)
        print(ill.shape)

        # compute mean of classes 
        exp_mean = np.reshape(np.mean(exp, axis=1), (exp.shape[0], 1))
        neutral_mean = np.reshape(np.mean(neutral, axis=1), (neutral.shape[0], 1))
        ill_mean = np.reshape(np.mean(ill, axis=1), (ill.shape[0], 1))

        # mean centered data
        exp = exp - np.tile(exp_mean, l/3)
        neutral = neutral - np.tile(neutral_mean, l/3)
        ill = ill - np.tile(ill_mean, l/3)

        # compute covariane matrix
        exp_cov = np.matmul(exp, exp.T)
        neutral_cov = np.matmul(neutral, neutral.T)
        ill_cov = np.matmul(ill, ill.T)

        # within class scatter matrix
        within_class_sm = 0.33333 * (exp_cov + neutral_cov + ill_cov)

        # overall mean
        mean = 0.33333 * (exp_mean + neutral_mean + ill_mean)
        print(within_class_sm)

        # between class scatter matrix
        m1 = exp_mean - mean
        m2 = neutral_mean - mean
        m3 = ill_mean - mean
        
        between_class_sm = 0.33333 * (np.matmul(m1,m1.T) + np.matmul(m2,m2.T) + np.matmul(m3, m3.T))

        print(between_class_sm.shape) 
        print(np.linalg.det(within_class_sm))

        if(np.linalg.det(within_class_sm) != 0):
            within_class_inv = np.linalg.inv(within_class_sm)
            J = np.linalg.trace(np.matmul(within_class_inv, between_class_sm))
            print(J.size)
        else:
            print("Determinent = 0 !!!!!!!!!")






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

# split_data(x['face'], 0)

# for key in y.keys():
#   print(key)

# for i in range(13):
#     im = x['face'][:,:,i]
#     print(im.shape)

#     cv2.imshow('image',im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# test()

PCA(x['face'])
# MDA(x['face'])