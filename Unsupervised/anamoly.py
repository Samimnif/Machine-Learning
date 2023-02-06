import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Drop the date column as it is not needed for unsupervised anamoly detection
df.drop('date', axis=1, inplace=True)

# Remove outliers
df = df[(np.abs(df - df.mean()) < (3 * df.std())).all(axis=1)]

# Check if there are any data points left after removing outliers
if df.shape[0] == 0:
    print("No data points left after removing outliers")
    exit()

# Plot histograms for each feature
df.hist(bins=10, figsize=(10,10))
plt.show()

# Plot heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# Plot scatter plots for each feature against every other feature
sns.pairplot(df, diag_kind='kde')
plt.show()

# Fit a Gaussian mixture model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(df)

# Predict the cluster for each data point
labels = gmm.predict(df)
df['labels'] = labels

# Plot the Gaussian mixture model
sns.scatterplot(x=df['meantemp'], y=df['humidity'], hue=df['labels'], palette='viridis')
plt.show()
