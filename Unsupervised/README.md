# Unsupervised Learning
## Clustering
In this experiment I used the Airbnb US 2020 data that includes teh following columns:
<br>
**Features:**

|                     |                   |                                |
|---------------------|-------------------|--------------------------------|
| id                  | latitude          | last_review                    |
| name                | longitude         | reviews_per_month              |
| host_id             | room_type         | calculated_host_listings_count |
| host_name           | price             | availability_365               |
| neighbourhood_group | minimum_nights    | city                           |
| neighbourhood       | number_of_reviews |                                |

### Objective
The objective of this experiment is to find the best model that predicts the price of the Airbnb rentals.
<br><br>
### Results
## Anamoly
In this experiment I used the wine quality dataset that includes the following
columns:
<br>
**Features:**

|                   |                    |
|-------------------|--------------------|
| fixedacidity      | totalsulfurdioxide | 
| volatileacidity   | density            |
| citricacid        | pH                 |
| residualsugar     | sulphates          |
| chlorides         | minimum_nights     |
| freesulfurdioxide | quality            |


### Objective
The objective of this is to generate models for wine quality dataset and find out the best one.
### Results
* I generated a heatmap of the trains and targets and got the following figure

![HeatMap](img/Classification-Figure_1.png)
* after splitting the data into trainign and tests and initializing the modele
```python
X = data[["fixedacidity","volatileacidity","citricacid","residualsugar","chlorides","freesulfurdioxide","totalsulfurdioxide","density","pH","sulphates","alcohol"]]
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```
![HeatMap](img/Classification-Figure_5.png)
```text
Accuracy: 0.7226027397260274
Confusion matrix:
 [[ 0  2  3  0  0]
 [ 0 98 26  3  0]
 [ 0 24 89  4  0]
 [ 0  5 12 23  0]
 [ 0  0  2  0  1]]
```
* Boxplot of the features:

![HeatMap](img/Classification-Figure_3.png)
* Scatter Plott

![HeatMap](img/Classification-Figure_4.png)


