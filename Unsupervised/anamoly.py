import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats

# Load the dataset
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# Convert any string columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean value of the column
df.fillna(df.mean(), inplace=True)

# Remove outliers using Z-score
X = df.values
z = np.abs(stats.zscore(X))
X = X[(z < 3).all(axis=1)]

if X.shape[0] == 0:
    print("No data points left after removing outliers.")
else:
    # Fit Isolation Forest on the data
    isolation_forest = IsolationForest(contamination=0.1)
    isolation_forest.fit(X)

    # Predict the anomaly score for each datapoint
    y_pred = isolation_forest.decision_function(X)
    y_pred = pd.Series(y_pred, name='Anomaly Score')

    # Plot the Gaussian Mixture Model (GMM)
    g = sns.jointplot(x=X[:, 0], y=X[:, 1], kind='kde')

    # Plot the scatter plot of the dataset with different colors for anomalies and inliers
    plt.scatter(X[y_pred > 0, 0], X[y_pred > 0, 1], s=100, c='blue', label='Inliers')
    plt.scatter(X[y_pred < 0, 0], X[y_pred < 0, 1], s=100, c='red', label='Anomalies')
    plt.title('Anomaly detection in Daily Delhi Climate Train data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # Plot the histogram for each feature
    df.hist(bins=20, figsize=(10, 10))
    plt.show()

    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True)
    plt.show()