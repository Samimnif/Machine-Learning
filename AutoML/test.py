import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Load the iris dataset
iris = load_iris()

# Convert the dataset to a pandas DataFrame
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Check for and remove any outliers
z_scores = stats.zscore(iris_df.iloc[:, :-1])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
iris_df = iris_df[filtered_entries]

# Histogram of the original dataset
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.histplot(iris_df['sepal length (cm)'], kde=True, ax=axs[0])
axs[0].set_title('Before Removing Outliers')

# Histogram of the filtered dataset
sns.histplot(iris_df['sepal length (cm)'], kde=True, ax=axs[1])
axs[1].set_title('After Removing Outliers')
plt.show()

# Boxplot of the filtered dataset
sns.boxplot(data=iris_df, orient='h')
plt.show()

# Gaussian plot of the filtered dataset
sns.kdeplot(iris_df['sepal length (cm)'])
plt.show()

# Heatmap of the correlation matrix
corr = iris_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Pairplot of the filtered dataset
sns.pairplot(iris_df, hue='target')
plt.show()
