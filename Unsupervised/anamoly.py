import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats

# Load the Daily Delhi Climate Train dataset
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# Convert any string columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean value of the column
df.fillna(df.mean(), inplace=True)


# Select the features and target
X = df.iloc[:, :5].values

# Remove outliers using Z-score
z = np.abs(stats.zscore(X))
X = X[(z < 3).all(axis=1)]

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

# Plot the heatmap of the Daily Delhi Climate Train dataset
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plot histograms of the features
df_features = df.iloc[:, :5]

plt.figure(figsize=(20, 12))
for i, col in enumerate(df_features.columns):
    plt.subplot(3, 2, i+1)
    plt.hist(df_features[col], color='blue', alpha=0.5)
    plt.title(col)
plt.tight_layout()
plt.show()
