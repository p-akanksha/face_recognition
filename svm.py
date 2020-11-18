# Support Vector Machine
# 
# Author: Akanksha Patel

class SVM:
    def __init__(self, train, test, kernel, c):
        self.train_data = train
        self.test_data = test
        self.kernel = kernel
        self.c = c


    def poly_kernel(x, y):
        res = np.matmul(x.T, y)**self.c
        return res

    def rbf(x, y):
        res = np.exp(-np.norm(x-y, axis=0)**2/self.c**2)
        return res

    
