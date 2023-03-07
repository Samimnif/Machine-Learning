import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv('cardio_train.csv', delimiter=';')

# Split the dataset into features and target
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier on the dataset
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Calculate the accuracy of the classifier
accuracy = rfc.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Save the trained model to a pickle file
pickle.dump(rfc, open("cardio_model.pkl", "wb"))

if __name__ == "__main__":
    # Plot a histogram of the age distribution
    plt.hist(df['age'], bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    # Plot a boxplot of the blood pressure distribution
    sns.boxplot(x=df['ap_hi'])
    plt.title("Blood Pressure Distribution")
    plt.xlabel("Blood Pressure")
    plt.show()

    # Plot a gaussian kde of the height distribution
    sns.kdeplot(x=df['height'], fill=True)
    plt.title("Height Distribution")
    plt.xlabel("Height")
    plt.show()

    # Plot a heatmap of the feature correlations
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Plot the AUC-ROC curve
    y_pred = rfc.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.show()

    # Plot a scatter plot of weight and height colored by cardio status
    sns.scatterplot(x='weight', y='height', hue='cardio', data=df)
    plt.title("Weight vs Height by Cardio Status")
    plt.show()

    # Plot pairplot
    sns.pairplot(data=df, hue='cardio')
    plt.title("Pairplot")
    plt.show()

    # Plot barplot of gender
    sns.countplot(data=df, x='gender', hue='cardio')
    plt.title("Gender Distribution")
    plt.show()