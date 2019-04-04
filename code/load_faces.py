import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from vis import grayscale_grid_vis


def load_faces( data_file = '../data/faces.txt'):
    X = np.loadtxt(data_file) / 255.
    X = np.reshape(np.transpose(np.reshape(X, (-1, 24, 24)), (0, 2, 1)), (len(X), -1))
    return X


X = load_faces()
grayscale_grid_vis(X[:64], (8,8), 'face_samples.png')
# print(X.shape)  # (4916, 576)

# question(a)
X_mean = np.mean(X, axis=0)
X_centralize = np.subtract(X, X_mean)
covariance_matrix = np.dot(X_centralize.T, X_centralize)/X.shape[0]
# print(covariance_matrix.shape)  # (576, 576)


# question(b)
values, vectors = linalg.eig(covariance_matrix)
vectors = np.transpose(vectors)

# v = np.dot(np.dot(vectors.T, covariance_matrix), vectors)
# vectors = np.reshape(np.transpose(np.reshape(vectors, (-1, 24, 24)), (0, 2, 1)), (len(vectors), -1))
grayscale_grid_vis(vectors[:64], (8,8), 'figure1.png')

# question(c)
u = vectors[:2]
components = np.dot(u, X.T)
plt.scatter(components[0][:1000], components[1][:1000])
# plt.show()
plt.savefig('figure2.png')

# question(d)
for k in range(len(values)):
    if np.sum(values[0:k]) >= 0.95*np.sum(values[:]):
        print(k)
        break

# question(e)
u = vectors[:8]
# print(u.shape)
components = np.dot(u, X.T)
x = np.argsort(components, axis=1)  # (8, 4916)
# print(x)
x = np.transpose(x)
images_index = np.transpose(x[x.shape[0]-8: x.shape[0]][:])
# print(images_index)
images = X[:64]
count = 0
for i in range(8):
    for j in range(8):
        images[count] = X[images_index[i][j]]
        count += 1

grayscale_grid_vis(u, (8,1), 'figure3.png')
grayscale_grid_vis(images, (8,8), 'figure4.png')
