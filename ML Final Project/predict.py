import pickle
import numpy as np
#import ML_CardioModel
#import ML_DiabetesModel

# Load the trained model
diabetes = pickle.load(open("diabetes_model.pkl", "rb"))
cardio = pickle.load(open("cardio_model.pkl", "rb"))

# Define a function to make predictions
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Convert the input to a numpy array
    input_data = np.array(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Make the prediction
    prediction = diabetes.predict(input_data)
    probability = diabetes.predict_proba(input_data)[0][1]

    # Return the prediction (0 or 1)
    return prediction[0], probability

# Define a function to make predictions
def predict_cardiovascular_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Convert the input to a numpy array
    input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

    # Make the prediction
    prediction = cardio.predict(input_data)
    probability = cardio.predict_proba(input_data)[0][1]

    # Return the prediction (0 or 1)
    return prediction[0], probability