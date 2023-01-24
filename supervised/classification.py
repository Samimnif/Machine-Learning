import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("winequality-red.csv")

encoder = LabelEncoder()
data["quality"] = encoder.fit_transform(data["quality"])

X = data[["fixedacidity","volatileacidity","citricacid","residualsugar","chlorides","freesulfurdioxide","totalsulfurdioxide","density","pH","sulphates","alcohol"]]
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matric:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu") #, cmap="Y1GnBu"
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("predictions")
plt.show()

sns.boxplot(data=X)
plt.show()

sns.pairplot(X, diag_kind='kde')
plt.show()