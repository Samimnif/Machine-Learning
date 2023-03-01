import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))


# Define a function to make predictions
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Convert the input to a numpy array
    input_data = np.array(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Return the prediction (0 or 1)
    return prediction[0]