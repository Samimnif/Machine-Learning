import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# Load the wine quality dataset
df = pd.read_csv('winequality-white.csv')

# Select the features and target
X = df.iloc[:, :10].values
y = df.iloc[:, -1].values

# Remove outliers using Z-score
z = np.abs(stats.zscore(X))
X = X[(z < 3).all(axis=1)]
y = y[(z < 3).all(axis=1)]

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

# Plot histograms of the features before and after removing outliers
df_features = df.iloc[:, :10]
df_features_no_outliers = pd.DataFrame(X, columns=df_features.columns)

plt.figure(figsize=(20, 12))
for i, col in enumerate(df_features.columns):
    plt.subplot(5, 2, i+1)
    plt.hist(df_features[col], color='blue', alpha=0.5, label='Before Outlier Removal')
    plt.hist(df_features_no_outliers[col], color='red', alpha=0.5, label='After Outlier Removal')
    plt.legend()
    plt.title(col)
plt.tight_layout()
plt.show()

