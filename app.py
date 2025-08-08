from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prediction function
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):

    # Directly parse form values (since HTML sends "1" or "0")
    male_encoded = int(male)
    currentSmoker_encoded = int(currentSmoker)
    BPMeds_encoded = int(BPMeds)
    prevalentStroke_encoded = int(prevalentStroke)
    prevalentHyp_encoded = int(prevalentHyp)
    diabetes_encoded = int(diabetes)

    # Prepare features
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded,
                          prevalentStroke_encoded, prevalentHyp_encoded, diabetes_encoded,
                          totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make prediction
    result = model.predict(scaled_features)
    # Optional: probability
    # probability = model.predict_proba(scaled_features)[0][1]

    return result[0]  # or return probability if using predict_proba

# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Collect and convert inputs
        male = request.form['male']
        age = int(request.form['age']) if request.form['age'] else 0
        currentSmoker = request.form['currentSmoker']
        cigsPerDay = float(request.form['cigsPerDay']) if request.form['cigsPerDay'] else 0.0
        BPMeds = request.form['BPMeds']
        prevalentStroke = request.form['prevalentStroke']
        prevalentHyp = request.form['prevalentHyp']
        diabetes = request.form['diabetes']
        totChol = float(request.form['totChol']) if request.form['totChol'] else 0.0
        sysBP = float(request.form['sysBP']) if request.form['sysBP'] else 0.0
        diaBP = float(request.form['diaBP']) if request.form['diaBP'] else 0.0
        BMI = float(request.form['BMI']) if request.form['BMI'] else 0.0
        heartRate = float(request.form['heartRate']) if request.form['heartRate'] else 0.0
        glucose = float(request.form['glucose']) if request.form['glucose'] else 0.0

        prediction = predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds,
                             prevalentStroke, prevalentHyp, diabetes,
                             totChol, sysBP, diaBP, BMI, heartRate, glucose)

        prediction_text = "The Patient has Heart Disease" if prediction == 1 else "The Patient has No Heart Disease"
        return render_template('index.html', prediction=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error in input: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port=3245)
