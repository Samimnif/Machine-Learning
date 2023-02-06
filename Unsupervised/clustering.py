import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the wine quality dataset
df = pd.read_csv('winequality-white.csv')

# Select the features and target
X = df.iloc[:, :10].values
y = df.iloc[:, -1].values

# Perform PCA to visualize the data in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit KMeans on the PCA-reduced data
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X_pca)

# Plot the Gaussian Mixture Model (GMM)
g = sns.jointplot(x=X_pca[:, 0], y=X_pca[:, 1], kind='kde')

# Plot the scatter plot of the clusters with each unique color
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of wine quality data')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Plot the heatmap of the wine quality dataset
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
