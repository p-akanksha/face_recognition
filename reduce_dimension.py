# Functions to reduce data dimensions
# 
# Author: Akanksha Patel

import cv2
import numpy as np
from sklearn import decomposition

def PCA(images):
    m, n, l = images.shape
    thresh = 0.97 # 0.99

    # Flatten Images
    flattened_images = []
    # To-do: change with reshape
    for i in range(l):
        im = np.ravel(images[:,:,i])
        flattened_images.append(im)

    # vectorized images in the columns 
    flattened_images = np.array(flattened_images)

    pca = decomposition.PCA(0.95).fit(flattened_images)
    components = pca.transform(flattened_images)  
    # x_recons = pca.inverse_transform(components)

    # # Make data zero-centric
    # mean = np.mean(flattened_images, axis=1)
    # mean = np.reshape(mean, (len(mean),1))
    # zere_cetered_x = flattened_images - np.tile(mean, l)

    # u, s, vh = np.linalg.svd(zere_cetered_x, full_matrices=True)

    # sm = np.sum(s) 
    
    # print(sm)
    # ss = 0
    # for i in range(s.shape[0]):
    #     ss = ss + s[i]
    #     if ss/sm > thresh:
    #         num = i
    #         break

    # print(num)

    # # sigma = np.zeros((num, vh.shape[0]))
    # # for i in range(min(vh.shape[0],num)):
    # #     sigma[i][i] = s[i]
    # # x_new = np.matmul(sigma, vh)

    # sigma = np.zeros((u.shape[0], vh.shape[0]))
    # for i in range(min(vh.shape[0], u.shape[0])):
    #     sigma[i][i] = s[i]
    # x_new = np.matmul(sigma, vh)

    # x_recons = np.matmul(u, x_new)
    # print(x_recons.shape)

    # for i in range(x_recons.shape[1]):
    #     img_flat = x_recons[:,i]

    #     # For vizualization
    #     img_min = np.min(img_flat)
    #     img_max = np.max(img_flat)
    #     print("Min, max: ", img_min, img_max)
    #     xx = img_max - img_min
    #     norm_factor = 1/xx
    #     img_min_arr = img_min * np.ones(img_flat.shape)
    #     img_flat = norm_factor * (img_flat - img_min_arr)
    #     print(img_flat.shape)

    #     img = np.reshape(img_flat, (m, n))

    #     cv2.imshow('image',img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     cv2.imshow('image',images[:,:,i])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # sm = np.sum(s) 

    # print(sm)
    # sh = s.shape
    # for i in range(sh[0]):
    #     # print((s[i]/sm)*100)
    #     if (s[i]/sm)*100 < thresh:
    #         print(i)
    #         break

    # print("PCA: ", u.shape)

    # x_new = []
    # u = []
    print("xxx", components.T.shape)
    print(pca.components_.T.shape)
    return components.T, pca.components_.T, pca # u[:,:num]

    
def MDA(images, data_type = 0):
    m, n, l = images.shape

    if data_type == 0: 
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