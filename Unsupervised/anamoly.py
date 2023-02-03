import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# Load the data
data = pd.read_csv("data.csv")

# Extract the features to be used in the anomaly detection algorithm
features = data.iloc[:, [2, 3, 4, 5]].values

# Fit the Gaussian Mixture Model to the data
gmm = GaussianMixture(n_components = 3)
gmm.fit(features)

# Predict the probabilities of each data point belonging to each cluster
probs = gmm.predict_proba(features)

# Compute the anomaly scores for each data point
scores = np.max(probs, axis = 1)

# Plot the Gaussian distributions for each cluster
for i in range(3):
    mean = gmm.means_[i,:]
    cov = gmm.covariances_[i,:]
    x, y = np.mgrid[mean[0]-2:mean[0]+2:.1, mean[1]-2:mean[1]+2:.1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(mean, cov)
    plt.contourf(x, y, rv.pdf(pos))

# Plot the scatter plot of the features, coloring each data point based on its anomaly score
plt.scatter(features[:, 0], features[:, 1], c = scores, cmap = 'viridis')

# Plot the heatmap of the features
sns.heatmap(data.corr(), annot = True)

# Plot the boxplot of the features, coloring each data point based on its anomaly score
sns.boxplot(x = "variable", y = "value", data = pd.melt(data[data.columns[2:6]], value_vars = data.columns[2:6]), hue = "value", hue_norm = (0, 1), palette = "viridis")

plt.show()
