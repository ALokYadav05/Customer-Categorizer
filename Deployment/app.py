from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r"src\best_model.pkl")  # Ensure this file exists in the same directory or provide full path

@app.route("/")
def home():
    return render_template("index.html")  # Renders the home page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Collect numerical and binary features from the form (21 features)
        input_data = [
            int(request.form["Age"]),
            int(request.form["Education"]),
            int(request.form["Parental_Status"]),
            int(request.form["Children"]),
            float(request.form["Income"]),
            float(request.form["Total_Spending"]),
            int(request.form["Days_as_Customer"]),
            int(request.form["Recency"]),
            int(request.form["Wines"]),
            int(request.form["Fruits"]),
            int(request.form["Meat"]),
            int(request.form["Fish"]),
            int(request.form["Sweets"]),
            float(request.form["Gold"]),
            int(request.form["Web"]),
            int(request.form["Catalog"]),
            int(request.form["Store"]),
            int(request.form["Web_Visits_Month"]),
            int(request.form["AcceptedCmp"]),
            int(request.form["Complain"]),
            int(request.form["Response"])
        ]

                # One-hot encode the Marital_Status (drop='first' was used while training, so 'Married' is dropped)
        marital_status = request.form["Marital_Status"]
        marital_status_categories = ['Together', 'Single', 'Divorced', 'Widow']  # Only 4, 'Married' is dropped
        marital_status_encoded = [1 if marital_status == cat else 0 for cat in marital_status_categories]


        # 3. Merge numerical + encoded features => Total = 25 features
        final_features = np.array(input_data + marital_status_encoded).reshape(1, -1)

        # 4. Make prediction
        prediction = model.predict(final_features)[0]

        return render_template("result.html", prediction_text=f"Predicted Customer Category: {prediction}")

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error : {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
