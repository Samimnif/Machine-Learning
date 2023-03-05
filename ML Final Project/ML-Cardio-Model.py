import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
