import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Show a sample of the dataset
print("Sample of the iris dataset:")
print(iris_df.sample(10))

# Histogram of all columns in the filtered dataset
iris_df.hist(figsize=(10, 10))
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_df.iloc[:, :-1], iris_df.iloc[:, -1],
                                                    test_size=0.3, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict the class of each iris in the testing set
y_pred = rfc.predict(X_test)

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

# Plot the predicted class of each iris in the testing set
fig, ax = plt.subplots()
sns.scatterplot(data=X_test, x='sepal length (cm)', y='sepal width (cm)', hue=y_pred, ax=ax)
ax.set_title('Predicted Class of Each Iris')
plt.show()