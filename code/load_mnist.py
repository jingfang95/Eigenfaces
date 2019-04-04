import sys
import os
import numpy as np
from time import time
from vis import grayscale_grid_vis
from numpy import linalg
import matplotlib.pyplot as plt


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
# print(X.shape)  # (1000,784)
grayscale_grid_vis(X[:64], (8, 8), '../mnist_samples.png')

# question(a)
X_mean = np.mean(X, axis=0)
X_centralize = np.subtract(X, X_mean)
covariance_matrix = np.dot(X_centralize.T, X_centralize)/X.shape[0]

values, vectors = linalg.eig(covariance_matrix)
vectors = np.transpose(vectors)
u = vectors[:2]
components = np.dot(u, X.T)
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}

for i in range(X.shape[0]):
    plt.scatter(components[0][i], components[1][i], c=colors[y[i]])

# plt.show()
plt.savefig('figure5.png')


