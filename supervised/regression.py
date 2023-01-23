'''
Linear Regression Model
'''

import matplotlib.pyplot as plt
from scipy import stats
import csv
import pandas as pd
import numpy as np
import seaborn as sns

rbnb = pd.read_csv("AB_US_2020.csv", low_memory=False)

numeric_col = ['price']
for x in ['price']:
    q75, q25 = np.percentile(rbnb.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)
    print(max, min)

    rbnb.loc[rbnb[x] < min, x] = np.nan
    rbnb.loc[rbnb[x] > max, x] = np.nan
rbnb = rbnb.dropna(axis=0)

'''"neighbourhood":[],"city":[],"room_type":[]'''
columns = {"minimum_nights": [], "number_of_reviews": [], "reviews_per_month": [], "availability_365": [],
           "calculated_host_listings_count": []}
prices = {"minimum_nights": [], "number_of_reviews": [], "reviews_per_month": [], "availability_365": [],
          "calculated_host_listings_count": []}
airbnbX = []
airbnbY = []
'''
with open('AB_US_2020.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            airbnbX.append(int(row[15]))
            airbnbY.append(int(row[9]))
        line_count += 1
'''
with open('AB_US_2020.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            for i in columns.keys():
                if row[i] != "" and float(row["price"]) > min and float(row["price"]) < max:
                    columns[i].append(float(row[i]))
                    prices[i].append(float(row["price"]))
        line_count += 1
bestR = -1
bestColumn = ""
for i in columns.keys():
    print(i)
    airbnbX = columns[i]
    airbnbY = prices[i]
    # print("X:",airbnbX)
    # print("Y:",airbnbY)
    slope, intercept, r, p, std_err = stats.linregress(airbnbX, airbnbY)
    print("Intercept:", intercept, "\nSlope:", slope, "\nRelation:", r, "\nFuntion: ", slope, " * x +", intercept)
    if i == list(columns.keys())[0]:
        bestR = r
        bestColumn = list(columns.keys())[0]
    else:
        if bestR < r:
            bestR = r
            bestColumn = i
    print("------------------------")
print("------------------------")
print("Best relation:", bestColumn, bestR)
airbnbX = columns[bestColumn]
airbnbY = prices[bestColumn]
slope, intercept, r, p, std_err = stats.linregress(airbnbX, airbnbY)
print("Intercept:", intercept, "\nSlope:", slope, "\nRelation:", r, "\nFuntion: ", slope, " * x +", intercept)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, airbnbX))

plt.scatter(airbnbX, airbnbY)
plt.plot(airbnbX, mymodel, color="red")
plt.xlabel(bestColumn.replace("_", " "))
plt.ylabel("Price")
plt.show()

boxplot = rbnb.boxplot(column=["minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365", "calculated_host_listings_count"], showfliers=False)
plt.show()

sns.heatmap(rbnb.corr(), annot=True)
plt.show()