import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the diabetes detection dataset
data = pd.read_csv("diabetes.csv")

# Split the dataset into training and testing data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = rfc.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model as a pickle file
pickle.dump(rfc, open("diabetes_model.pkl", "wb"))

if __name__ == "__main__":
    # Plot feature importance
    feat_importances = pd.Series(rfc.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    y_pred = rfc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
