from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('leukemia_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from your Noir Gold HTML
        # Ensure the order is: [Age, WBC, RBC, Platelets, Hb, BMI]
        features_list = [
            float(request.form['age']),
            float(request.form['wbc']),
            float(request.form['rbc']),
            float(request.form['platelets']),
            float(request.form['hb']),
            float(request.form['bmi'])
        ]
        
        final_features = np.array([features_list])
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)

        # THE INVERSION FIX:
        # If prediction is 1, but it's showing for normal values, 
        # swap the 1 and 0 logic below.
        if prediction[0] == 0: 
            result = "Positive: Leukemia Signals Detected"
        else:
            result = "Negative: No Significant Anomalies"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)