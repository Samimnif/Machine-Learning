from flask import Flask, render_template, request
import numpy as np

# Load the predict_diabetes function
from predict import predict_diabetes

# Initialize the Flask app
app = Flask(__name__)


# Define the home page
@app.route("/")
def home():
    return render_template("home.html")


# Define the prediction page
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input from the form
    Pregnancies = int(request.form["Pregnancies"])
    Glucose = int(request.form["Glucose"])
    BloodPressure = int(request.form["BloodPressure"])
    SkinThickness = int(request.form["SkinThickness"])
    Insulin = int(request.form["Insulin"])
    BMI = float(request.form["BMI"])
    DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
    Age = int(request.form["Age"])

    # Make the prediction using the predict_diabetes function
    prediction = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                  DiabetesPedigreeFunction, Age)

    # Return the prediction to the user
    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
