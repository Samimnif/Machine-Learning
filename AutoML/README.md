# AutoML
## Objective
### Dataset
```python
     sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
80                 5.5               2.4  ...               1.1     1.0
0                  5.1               3.5  ...               0.2     0.0
63                 6.1               2.9  ...               1.4     1.0
66                 5.6               3.0  ...               1.5     1.0
14                 5.8               4.0  ...               0.2     0.0
104                6.5               3.0  ...               2.2     2.0
122                7.7               2.8  ...               2.0     2.0
130                7.4               2.8  ...               1.9     2.0
30                 4.8               3.1  ...               0.2     0.0
75                 6.6               3.0  ...               1.4     1.0
```
### Explainability
The model is an AutoML classifier for the Iris dataset, which means it automatically trains and selects the best classification algorithm for the task of predicting the species of an iris plant based on its sepal length, sepal width, petal length, and petal width.

The AutoML model uses the scikit-learn library's train_test_split function to split the dataset into training and testing sets, and then the LazyClassifier function from the mlxtend library to train and evaluate multiple classification algorithms on the training set. The LazyClassifier function automatically selects the best algorithm for the task based on the accuracy score on the training set.

After the best algorithm is selected, the model trains the algorithm on the entire dataset and uses it to make predictions on the test set. Finally, the model outputs the accuracy score of the selected algorithm on the test set and a confusion matrix to show the number of true positives, false positives, true negatives, and false negatives for each class.