from scipy.misc import imsave
import numpy as np
import math

def grayscale_grid_vis(X, (nh, nw), save_path=None):
    """
        Inputs:
          x: shape=(n, d), n: number of images, d: feature dimension
          (nh, nw): plot n images in a nh x nw grid
    """
    assert X.ndim == 2, 'illeal inputs'
    dim = int(math.sqrt(X[0].shape[-1]))
    X = np.reshape(X, (-1, dim, dim))

    h, w = X[0].shape[:2] # image width and height 

    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


