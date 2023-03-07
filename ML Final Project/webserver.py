from flask import Flask, render_template, request
import numpy as np

# Load the predict_diabetes function
from predict import predict_diabetes , predict_cardiovascular_disease

# Initialize the Flask app
app = Flask(__name__)


# Define the home page
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/cardio")
def cardio():
    return render_template("cardio.html")


# Define the prediction page
@app.route("/predict_diabetes", methods=["POST"])
def predict_d():
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
    return render_template("predict.html", prediction=prediction[0], probability=prediction[1])

@app.route("/predict_cardio", methods=["POST"])
def predict_c():
    # Get the input from the form
    gender = request.form.get("gender")
    height = int(request.form["height"])
    weight = int(request.form["weight"])
    ap_hi = int(request.form["ap_hi"])
    ap_lo = int(request.form["ap_lo"])
    cholesterol = request.form.get("cholesterol")
    gluc = request.form.get("gluc")
    smoke = request.form.get("smoke")
    active = request.form.get("active") #add this
    alco = request.form.get("alco") #add this
    age = int(request.form["age"]) * 365

    # Make the prediction using the predict_diabetes function
    prediction = predict_cardiovascular_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)

    # Return the prediction to the user
    return render_template("predict.html", prediction=prediction[0], probability=prediction[1])


if __name__ == "__main__":
    app.run(debug=True)
