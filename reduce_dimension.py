# Functions to reduce data dimensions
# 
# Author: Akanksha Patel

import cv2
import numpy as np
from sklearn import decomposition
from sklearn.covariance import LedoitWolf

def PCA(images, thresh):
    m, n, l = images.shape
    # print(images.shape)
    # m, l = images.shape # 0.99
    print(thresh)

    # sklearn pca for comparision
    # pca = decomposition.PCA(0.95).fit(images.T)
    # components = pca.transform(images.T)  
    # x_recons = pca.inverse_transform(components)

    # # flatten_images
    data = []
    data_labels = []
    print("xxxxx", l, n)
    for i in range(l):
        for j in range(n):
            data.append(images[:,j,i])
            data_labels.append(i)

    data = np.asarray(data).T
    data_labels = np.asarray(data_labels).T

    mean = np.mean(data, axis=1)
    mean = np.reshape(mean, (len(mean), 1))
    zero_cetered_x = data - np.tile(mean, l*n)
    # cov_data = np.matmul(zero_cetered_x, zero_cetered_x.T)

    # eig_value, eig_vector = np.linalg.eig(cov_data)

    # # Sorting eigen value and corresponding eig vector in descending order
    # eig_value = eig_value.real
    # eig_value = eig_value / np.sum(eig_value)
    # index = np.argsort(eig_value)
    # index = index[::-1]
    # eig_value_sort = eig_value[index]

    # # Calculate the number of features to use
    # ss = 0
    # for i in range(len(eig_value_sort)):
    #     ss = ss + eig_value_sort[i]
    #     if ss > thresh:
    #         num = i
    #         break

    # eig_vector_sort = eig_vector[:, index[:num]]

    # # Taking only real values
    # eig_vector_sort = eig_vector_sort.real

    # principle_components = np.matmul(zero_cetered_x.T, eig_vector_sort).T

    # print(principle_components.shape)

    # return principle_components

    # covar = np.dot(imgDiff.transpose(), imgDiff)/float(img.shape[1] - 1)
    # EigVal, EigVec = np.linalg.eigh(covar)
    # idx = np.argsort(-EigVal)
    # EigVal = EigVal[idx]
    # EigVec = EigVec[:,idx]
    # SigEigVec = EigVec[:,:2]
    # pca_vec = np.dot(imgDiff,SigEigVec)
    # pca_vec = pca_vec.astype(np.float)
    # return pca_vec


    # # Make data zero-centric
    # mean = np.mean(data, axis=1)
    # mean = np.reshape(mean, (len(mean), 1))
    # zero_cetered_x = data - np.tile(mean, l*n)

    u, s, vh = np.linalg.svd(zero_cetered_x, full_matrices=True)

    sm = np.sum(s) 
    
    ss = 0
    for i in range(s.shape[0]):
        ss = ss + s[i]
        if ss/sm > thresh:
            num = i
            break

    print("Using " + str(num) + " features out of " + str(m))

    sigma = np.zeros((num, vh.shape[0]))
    for i in range(min(vh.shape[0],num)):
        sigma[i][i] = s[i]
    x_new = np.matmul(sigma, vh)

    # For reconstruction
    # sigma2 = np.zeros((u.shape[0], vh.shape[0]))
    # for i in range(min(vh.shape[0], u.shape[0])):
    #     sigma2[i][i] = s[i]
    # x_new2 = np.matmul(sigma2, vh)

    # x_recons = np.matmul(u, x_new2)
    # print(x_recons.shape)

    # for i in range(x_recons.shape[1]):
    #     img_flat = x_recons[:,i]
    #     print(img_flat)

    #     # For vizualization
    #     mm = np.mean(img_flat)
    #     print(mm)
    #     img_min = np.min(img_flat)
    #     img_max = np.max(img_flat)
    #     print("Min, max: ", img_min, img_max)
    #     xx = img_max - img_min
    #     norm_factor = 1/xx
    #     img_min_arr = img_min * np.ones(img_flat.shape)
    #     img_flat = norm_factor * (img_flat - img_min_arr)
    #     print(img_flat)

    #     img = np.reshape(img_flat, (24, 21))

    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     cv2.imshow('image', np.reshape(data[:,i], (24, 21)))
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

    return x_new, data_labels

    # return components.T, pca.components_.T, pca 

    
def MDA(data, n_class, num):
    m, n, l = data.shape

    # train = np.zeros((m, n*l))
    # train_labels = np.zeros((n*l, 1))

    # for i in range(n):
    #     for j in range(l):
    #         train[:,l*i+j] = data[:,i,j]
    #         train_labels[l*i+j,:] = i

    # data_class = []

    # for i in range(n_class):
    #     cl = []
    #     for j in range(n*l):
    #        :ic train_labels[j] == i:
    #             cl.append(train[:,j])
    #     if cl
    #     data_class.append(np.asarray(cl))

    # data_class = np.asarray(data_class)
    # data = :pczeros((m, l/n_class, n_class))
    # print(data_class.shape)
    # print(data_class_2.shape)
    # for i in range(l/n_class):
    #     for j in range(n_class):
    #         data_class_2[:,i,j] = data_class[j,i,:]

    # compute mean of classes 
    mean = np.zeros((m, 1, n_class))
    for c in range(n_class):
        d = data[:,:,c]
        mean[:, :, c] = np.reshape(np.mean(d, axis=1), (d.shape[0], 1))

    # covariance of different features
    cov = np.zeros((m, m, n_class))
    for c in range(n_class):
        d = data[:,:,c].T
        sig = LedoitWolf().fit(d).covariance_
        X = np.random.multivariate_normal(mean=mean[:,:,c].squeeze(),
                                              cov=sig,
                                              size=50)
        cov[:,:,c] = LedoitWolf().fit(d).covariance_

    # within class scatter matrix
    within_class_sm = np.zeros_like(cov[:,:,0])
    for c in range(n_class):
        within_class_sm = within_class_sm + cov[:,:,c]

    within_class_sm = 1/float(n_class) * within_class_sm

    # overall mean
    overall_mean = np.zeros_like(mean[:,:,0])
    for c in range(n_class):
        overall_mean = overall_mean + mean[:,:,c]

    overall_mean = 1/float(n_class) * overall_mean

    print(within_class_sm)

    # between class scatter matrix
    between_class_sm = np.zeros_like(np.matmul(mean[:,:,0],mean[:,:,0].T))
    for c in range(n_class):
        between_class_sm = between_class_sm + np.matmul(mean[:,:,c],mean[:,:,c].T)

    between_class_sm = 1/float(n_class) * between_class_sm

    # m1 = exp_mean - mean
    # m2 = neutral_mean - mean
    # m3 = ill_mean - mean
    
    # between_class_sm = 0.33333 * (np.matmul(m1,m1.T) + np.matmul(m2,m2.T) + np.matmul(m3, m3.T))

    print(between_class_sm.shape) 
    print(np.linalg.det(within_class_sm))

    if(np.linalg.det(within_class_sm) != 0):
        within_class_inv = np.linalg.inv(within_class_sm)
        J = np.matmul(np.linalg.inv(within_class_sm), between_class_sm)
        print(J.size)
    else:
        print("Determinent = 0 !!!!!!!!!")

    eig_value, eig_vector = np.linalg.eig(J)

    # Sorting eigen value and corresponding eig vector in descending order
    eig_value = eig_value.real
    eig_value = eig_value / np.sum(eig_value)
    index = np.argsort(eig_value)
    index = index[::-1]
    eig_value_sort = eig_value[index]
    eig_vector_sort = eig_vector[:, index]

    # Taking first num components 
    eig_vector_sort = eig_vector_sort.real[:,:num]

    X = np.zeros(m, n*l)
    X_labels = np.zeros((n*l, 1))
    print("xxx", l, n)
    for i in range(l):
        for j in range(n):
            X[:,n*i+j] = data[:,i,j]
            X_labels[n*i+j,:] = i

    X_new = np.matmul(X.T, eig_vector_sort)

    return X_new, X_labels 
    # variance = np.std(principle_components, axis=0)
    # index = np.argsort(variance)

    # print(mean.shape)


        # exp = []
        # neutral = []
        # ill = []

        # for i in range(0, l, 3):
        #     exp.append(np.ravel(images[:,:,i]))
        #     neutral.append(np.ravel(images[:,:,i+1]))
        #     ill.append(np.ravel(images[:,:,i+2]))

        # # convert to np array
        # exp = np.array(exp).T
        # neutral = np.array(neutral).T
        # ill = np.array(ill).T

        # print(exp.shape)
        # print(neutral.shape)
        # print(ill.shape)

        # # compute mean of classes 
        # exp_mean = np.reshape(np.mean(exp, axis=1), (exp.shape[0], 1))
        # neutral_mean = np.reshape(np.mean(neutral, axis=1), (neutral.shape[0], 1))
        # ill_mean = np.reshape(np.mean(ill, axis=1), (ill.shape[0], 1))

        # # mean centered data
        # exp = exp - np.tile(exp_mean, l/3)
        # neutral = neutral - np.tile(neutral_mean, l/3)
        # ill = ill - np.tile(ill_mean, l/3)

        # # compute covariane matrix
        # exp_cov = np.matmul(exp, exp.T)
        # neutral_cov = np.matmul(neutral, neutral.T)
        # ill_cov = np.matmul(ill, ill.T)

        # # within class scatter matrix
        # within_class_sm = 0.33333 * (exp_cov + neutral_cov + ill_cov)

        # # overall mean
        # mean = 0.33333 * (exp_mean + neutral_mean + ill_mean)
        # print(within_class_sm)

        # # between class scatter matrix
        # m1 = exp_mean - mean
        # m2 = neutral_mean - mean
        # m3 = ill_mean - mean
        
        # between_class_sm = 0.33333 * (np.matmul(m1,m1.T) + np.matmul(m2,m2.T) + np.matmul(m3, m3.T))

        # print(between_class_sm.shape) 
        # print(np.linalg.det(within_class_sm))

        # if(np.linalg.det(within_class_sm) != 0):
        #     within_class_inv = np.linalg.inv(within_class_sm)
        #     J = np.linalg.trace(np.matmul(within_class_inv, between_class_sm))
        #     print(J.size)
        # else:
        #     print("Determinent = 0 !!!!!!!!!")