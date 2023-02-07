import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# Load the dataset
data = pd.read_csv('DailyDelhiClimateTrain.csv')

# Remove outliers using the Z-Score method
z_scores = np.abs(zscore(data.drop('date', axis=1)))
data = data[(z_scores < 3).all(axis=1)]

# Print a sample table of the data
print(data.head())

# Plot a heatmap to visualize the correlation between features
sns.heatmap(data.corr(), annot=True)
plt.show()

# Plot a boxplot to visualize the distribution of each feature
sns.boxplot(data=data)
plt.show()

# Plot a histogram to visualize the distribution of each feature
data.hist()
plt.show()

# Fit an Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.05)
model.fit(data.drop('date', axis=1))

# Predict anomalies and add the predictions to the dataframe
anomalies = model.predict(data.drop('date', axis=1))
data['anomaly'] = anomalies

# Plot a scatter plot to visualize the anomalies
sns.scatterplot(x='meanpressure', y='meantemp', data=data, hue='anomaly', palette={1:'blue', -1:'red'})
plt.show()

# Plot a Gaussian distribution to visualize the distribution of anomalies
sns.distplot(data[data.anomaly==1].meantemp, kde=False, color='blue', label='Normal')
sns.distplot(data[data.anomaly==-1].meantemp, kde=False, color='red', label='Anomaly')
plt.legend()
plt.show()
