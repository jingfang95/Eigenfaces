import sys
import os
import numpy as np
from time import time
from vis import grayscale_grid_vis
from numpy import linalg
import matplotlib.pyplot as plt
from k_means_evaluation import  accuracy_score

def kmeans(X, k = 3, max_iter = 100, random_state=0):
    """
    Inputs:
        X: input data matrix, numpy array with shape (n * d), n: number of data points, d: feature dimension
        k: number of clusters
        max_iters: maximum iterations
    Output:
        clustering label for each data point
    """
    assert len(X) > k, 'illegal inputs'
    np.random.seed(random_state)

    # randomly select k data points as center
    idx = np.random.choice(len(X), k, replace=False)
    print(idx-180)
    centers = X[idx-180]

    # please complete the following code: 

    from scipy.spatial import distance
    for i in range(max_iter):

        # Update labels of each data point
        H = distance.cdist(X, centers, 'euclidean') # calculate the distance between data and centers
        labels = np.argsort(H, axis=1) # find the label of each data points (the length of labels = n)
        # use argsort to avoid for loop
        labels = np.transpose(labels)[0]

        # Update centers of each cluster
        for c in range(k):
            count = 0
            for i in range(X.shape[0]):
                if labels[i] == c:
                    count += 1
            subset = X[:count]
            count = 0
            for i in range(X.shape[0]):
                if labels[i] == c:
                    subset[count] = X[i]  # find the subset of points whose label equals c
                    count += 1
            centers[c] = np.mean(subset, axis=0)  # calculate the center of cluster c.

    return labels


def load_mnist( data_dir = '../data/mnist_data'):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    trY = np.asarray(trY)

    X = trX / 255.
    y = trY

    subset  = [i for i, t in enumerate(y) if t in [1, 0, 2, 3]]
    X, y = X.astype('float32')[subset], y[subset]
    return X[:1000], y[:1000]

X, y = load_mnist()
y_true = y
X_mean = np.mean(X, axis=0)
X_centralize = np.subtract(X, X_mean)
covariance_matrix = np.dot(X_centralize.T, X_centralize)/X.shape[0]

values, vectors = linalg.eig(covariance_matrix)
vectors = np.transpose(vectors)
u = vectors[:2]
components = np.dot(u, X.T)
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}

# question(c, d)
y = kmeans(X, k=4)
for i in range(X.shape[0]):
    plt.scatter(components[0][i], components[1][i], c=colors[y[i]])
score = accuracy_score(y_true, y)
print(score)

# plt.show()
# plt.savefig('figure10.png')


# question(e)
u = vectors[:784]
components = np.dot(u, X.T)

y = kmeans(X, k=4)
for i in range(X.shape[0]):
    plt.scatter(components[0][i], components[1][i], c=colors[y[i]])

# plt.savefig('figure11.png')
score = accuracy_score(y_true, y)
print(score)
# plt.show()
