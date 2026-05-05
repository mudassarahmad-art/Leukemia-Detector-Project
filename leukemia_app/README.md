# Leukemia Risk Detector — Flask App

A modern, minimalist Flask web app that estimates leukemia risk percentage from a clinical questionnaire and visualizes the result on a green → yellow → red gradient bar.

**Theme:** Noir & Gold (premium black + gold accents).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Plug in your trained ML model

1. Save your trained classifier as `model/leukemia_model.pkl` (use `joblib.dump(model, "model/leukemia_model.pkl")`).
2. Make sure your model was trained on **the same feature order and encoding** defined in `app.py`:
   - Order: `wbc, rbc, platelets, blasts, age, gender, bmi, smoker, alcohol, family_history, chronic_illness`
   - Encoding: see the `FEATURE_ENCODING` dict in `app.py` — edit it to match your training pipeline.
3. The app will:
   - call `model.predict_proba(X)[0][1]` if available (classification), or
   - call `model.predict(X)[0]` (regression — auto-scaled to 0–100).
4. If no model is found, the app falls back to a transparent rule-based weighted score so the UI still works for your demo.

## Project structure

```
leukemia_app/
├── app.py
├── requirements.txt
├── model/
│   └── leukemia_model.pkl   ← put your trained model here
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```
