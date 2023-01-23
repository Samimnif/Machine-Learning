import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("bank-full.csv")
#,"volatileacidity","citricacid","residualsugar","chlorides","freesulfurdioxide","totalsulfurdioxide","density","pH","sulphates","alcohol"
x = data.drop("balance", axis=1)
y = data["age"]

le = LabelEncoder()
le.fit(data[""])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("true labels")
plt.ylabel("Predicted labels")
plt.title("Scatter plot of predicted vs actual labels")
plt.show()

sns.heatmap(x.corr(), annot=True)
plt.show()