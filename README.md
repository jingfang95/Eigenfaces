# Eigenfaces

Applied principal component analysis to face detection by centralizing the features from a covariance matrix (576x576) and plotting the "eigenfaces" associated with the top 64 largest eigenvalues.

Explaination:
Principal component analysis (PCA) is a powerful tool for dimension reduction, which projects high-dimensional data points to low-dimensional feature space while preserving most of the important information of the original data. Here, I explored this on a dataset of face images. This dataset consists of a collection of 24x24 face images, which are flattened into 576(= 24x24)-dimensional vectors, with each pixel treated as a feature.

1. Centralize the features (pixels) to make them have zero mean across the dataset. I did this by subtracting each image by the mean of all the images, then calculated the covariance matrix.
2. Calculated the eigenvectors and eigenvalues of the covariance matrix. Each eigenvector has the same dimensionality (=576) as the original images, and thus I could reshape them and viewed them as “face images” similar to the original data (use vis.py). The eigenvectors of this covariance matrix are therefore called eigenfaces. I plotted the eigenfaces associated with the top 64 largest eigenvalues in a 8x8 grid using vis.py.
3. Performed dimension reduction by projecting the data onto the top k eigenvectors. For visualization purpose, I considered reducing the dimension to k = 2. 
4. Using only the two eigenvectors may loss too much information, then I decided a threshold k, such that the top k eigenvectors include a large portion of the variance. Since each of the eigenvalues denotes the variance of the corresponding principal component, it is useful to visualize the eigenvalues by plotting l vs. λ, where λ denotes the l-th largest eigenvalue.
5. Each of the eigenvector may capture a different aspect of the data. To have a visualization, I took the top 8 eigenvectors, found the top 8 images whose project is maximized, and used vis.py to visualize both the eigenface and its top 8 images.
