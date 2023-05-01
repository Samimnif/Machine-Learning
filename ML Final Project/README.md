# ML Final Project
## Objective
The objective of this project is to practice my ML skills and apply what I learned.
I will be using real-world data to predict medical conditions like cardiovascular disease and Diabetes type II.
## libraries
```python
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
```
## Datasets
## Structure
![structure](static/images/structure.png)
![directory](static/images/dir.png)
## Analysis
### Diabetes
![feature_Diabetes](static/images/diabetes_feature.png)
![histogram_dia](static/images/diabetes_histogram.png)
![confusion_dia](static/images/diabetes_confusion.png)
![density_dia](static/images/diabetes_density.png)
![boxplot_dia](static/images/diabetes_boxplot.png)
### Cardiovascular 
![histogram_card](static/images/cardio_histogram.png)
![heatmap_card](static/images/cardio_heatmap.png)
![boxplot_card](static/images/cardio_boxplot.png)
![barplot_card](static/images/cardio_barplot.png)
![AUC-ROC_card](static/images/cardio_AUC-ROC.png)
![gaussian_card](static/images/cardio_gaussian.png)
![scatter_card](static/images/cardio_scatter.png)
![pairplot_card](static/images/cardio_pairplot.png)
