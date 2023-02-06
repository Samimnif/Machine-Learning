import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load the data
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# drop the date column
df = df.drop('date', axis=1)

# convert the data into a numpy array
X = df.values

# remove outliers using Z-score
z = np.abs(stats.zscore(X))
X = X[(z < 3).all(axis=1)]

# check if there are any data points left after removing outliers
if X.shape[0] == 0:
    print("No data points left after removing outliers")
else:
    # calculate the covariance matrix
    covariance_matrix = np.cov(X.T)

    # eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # calculate Mahalanobis distance
    mahalanobis_distance = X.dot(np.linalg.inv(covariance_matrix)).dot((X - np.mean(X, axis=0)).T)
    mahalanobis_distance = np.diag(mahalanobis_distance)

    # plot the histogram of the Mahalanobis distance
    plt.hist(mahalanobis_distance, bins=100)
    plt.title("Histogram of Mahalanobis Distance")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.show()

    # plot the scatter plot of the data
    sns.scatterplot(df.index, mahalanobis_distance)
    plt.title("Scatter Plot of Mahalanobis Distance")
    plt.xlabel("Data Point Index")
    plt.ylabel("Mahalanobis Distance")
    plt.show()

    # plot the Gaussian plot of the Mahalanobis distance
    sns.distplot(mahalanobis_distance, fit=stats.norm, kde=False)
    plt.title("Gaussian Plot of Mahalanobis Distance")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.show()

    # plot the heatmap of the covariance matrix
    sns.heatmap(covariance_matrix, cmap="coolwarm")
    plt.title("Heatmap of Covariance Matrix")
    plt.show()
