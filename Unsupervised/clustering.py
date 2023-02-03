import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

# Load the data
data = pd.read_csv("data.csv")

# Extract the features and target to be used in the clustering algorithm
features = data.iloc[:, [2, 3, 4, 5]].values
target = data.iloc[:, 6].values

# Fit the K-Means algorithm to the data
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

# Plot the Gaussian distributions for each cluster
for i in range(3):
    mean = kmeans.cluster_centers_[i,:]
    cov = np.cov(features[y_kmeans == i].T)
    x, y = np.mgrid[mean[0]-2:mean[0]+2:.1, mean[1]-2:mean[1]+2:.1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(mean, cov)
    plt.contourf(x, y, rv.pdf(pos))

# Plot the heatmap of the features
sns.heatmap(data.corr(), annot = True)

# Plot the data with different colors for each cluster
plt.scatter(features[y_kmeans == 0, 0], features[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[y_kmeans == 1, 0], features[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(features[y_kmeans == 2, 0], features[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

# Plot the centroids of each cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
