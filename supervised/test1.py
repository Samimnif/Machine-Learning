import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
plt.scatter(x, y)
plt.show()

print (x,y)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
print("----------------------")
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print (x,y)
print(f"intercept: {new_model.intercept_}")
print(f"slope: {new_model.coef_}")
print("----------------------")
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")