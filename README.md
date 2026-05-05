
Copy

# 🩺 Leukemia Prediction — ML Classifier
 
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green?style=flat-square)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
 
> Predicts leukemia status (**Positive / Negative**) from routine blood-test values using a Random Forest classifier with SMOTE-based class balancing.
 
---
 
## 📊 Project Snapshot
 
| Property | Detail |
|---|---|
| **Model** | Random Forest (`n_estimators=100`) |
| **Balancing** | SMOTE (training set only) |
| **Train / Test split** | 80 / 20 stratified |
| **Input features** | 6 clinical blood-test values |
| **Target** | `Leukemia_Status` → Positive / Negative |
| **Export** | `leukemia_model.pkl` + `scaler.pkl` (Flask-ready) |
 
---
 
## 🗂 Table of Contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Set](#feature-set)
- [Class Imbalance & SMOTE](#class-imbalance--smote)
- [Pipeline](#pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Export & Deployment](#export--deployment)
- [Disclaimer](#disclaimer)
---
 
## Overview
 
Leukemia datasets suffer from **severe class imbalance** — far fewer positive cases than negative ones. A naive model simply predicts "Negative" for everything and still scores high accuracy, missing every real case.
 
This project solves that end-to-end:
 
1. Load and clean the biased dataset (fix erroneous negative WBC values)
2. Run EDA — class distribution plots, box plots, correlation heatmap
3. Apply **SMOTE** to the training set only (no data leakage)
4. Scale with `StandardScaler` (fit on train, transform both)
5. Train a **Random Forest** on the balanced data
6. Evaluate on the original unbalanced test set
7. Export model + scaler as `.pkl` files for deployment
---
 
## Dataset
 
| Property | Detail |
|---|---|
| **File** | `biased_leukemia_dataset.csv` |
| **Target column** | `Leukemia_Status` (`Positive` / `Negative`) |
| **Known issue** | Severe class imbalance (~85% Negative) |
| **Cleaning** | `WBC_Count` absolute-valued to remove erroneous negatives |
 
---
 
## Feature Set
 
| # | Feature | Category | Clinical Meaning | Relative Importance |
|---|---|---|---|---|
| 1 | `WBC_Count` | 🔴 Primary | White blood cell count — key leukemia marker | `████████████` High |
| 2 | `Hemoglobin_Level` | 🟢 Blood | Oxygen-carrying protein level | `██████████` Med-High |
| 3 | `Platelet_Count` | 🟢 Blood | Clotting cells — often low in leukemia | `█████████` Medium |
| 4 | `RBC_Count` | 🟢 Blood | Red blood cell count | `███████` Medium |
| 5 | `Age` | 🔵 Demo | Patient age | `█████` Low-Med |
| 6 | `BMI` | 🔵 Demo | Body Mass Index | `███` Low |
 
---
 
## Class Imbalance & SMOTE
 
### Before SMOTE — severely imbalanced
 
```
Negative  ██████████████████████████████████████████  ~85%
Positive  ██████                                       ~15%
```
 
A model trained here learns to always predict **Negative** — high accuracy, zero clinical value.
 
### After SMOTE — perfectly balanced
 
```
Negative  █████████████████████  50%
Positive  █████████████████████  50%
```
 
SMOTE generates **synthetic minority samples** by interpolating between existing positive-class neighbors. Applied **only on the training set** to prevent data leakage into evaluation.
 
### Key rule: no leakage
 
```
Raw data
   ├── Train set  →  SMOTE applied here  →  balanced train
   └── Test set   →  untouched            →  real-world distribution
```
 
---
 
## Pipeline
 
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Load CSV   │────▶│     EDA      │────▶│ 80/20 Split  │────▶│    SMOTE     │
│  clean WBC   │     │ plots/heatmap│     │  stratified  │     │ train only   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                                                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Export     │◀────│   Evaluate   │◀────│  RF n=100    │◀────│ StandardScale│
│  .pkl files  │     │ report + CM  │     │    train     │     │ fit on train │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```
 
---
 
## Evaluation
 
### Confusion Matrix structure
 
```
                  Predicted Negative    Predicted Positive
Actual Negative  ┌─────────────────┐  ┌─────────────────┐
                 │   True Negative  │  │  False Positive  │
                 └─────────────────┘  └─────────────────┘
Actual Positive  ┌─────────────────┐  ┌─────────────────┐
                 │  False Negative  │  │   True Positive  │
                 └─────────────────┘  └─────────────────┘
```
 
### Metrics
 
| Metric | Formula | What it means |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of predicted positives, how many were correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were caught |
| **F1-Score** | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| **Accuracy** | (TP + TN) / Total | Overall correctness — misleading on imbalanced data |
 
> In medical ML, **Recall (sensitivity)** is most critical — missing a positive case has higher cost than a false alarm.
 
---
 

 
## Getting Started
 
### 1. Clone the repo
 
```bash
git clone https://github.com/your-username/leukemia-detection.git
cd leukemia-detection
```
 
### 2. Install dependencies
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```
 
Or via requirements file:
 
```bash
pip install -r requirements.txt
```
 
### 3. Add the dataset
 
Place `biased_leukemia_dataset.csv` in the project root.
 
### 4. Run the notebook
 
```bash
jupyter notebook Leukemia_prediction_Project1_1.ipynb
```
 
Run all cells top to bottom. The final cell saves `leukemia_model.pkl` and `scaler.pkl`.
 
---
 
## Export & Deployment
 
The notebook exports two files for use in a Flask (or any Python) web app:
 
| File | Contents |
|---|---|
| `leukemia_model.pkl` | Trained `RandomForestClassifier` |
| `scaler.pkl` | Fitted `StandardScaler` |
 
### Usage example
 
```python
import joblib
import numpy as np
 
model  = joblib.load('leukemia_model.pkl')
scaler = joblib.load('scaler.pkl')
 
# Input: [Age, WBC_Count, RBC_Count, Platelet_Count, Hemoglobin_Level, BMI]
sample = np.array([[45, 11000, 4.5, 250000, 13.5, 24.0]])
 
prediction = model.predict(scaler.transform(sample))
print("Positive" if prediction[0] == 1 else "Negative")
```
 
---
 
## Dependencies
 
| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `numpy` | ≥ 1.21 | Numerical operations |
| `matplotlib` | ≥ 3.4 | Plotting |
| `seaborn` | ≥ 0.11 | Statistical visualisation |
| `scikit-learn` | ≥ 1.0 | Model, scaling, evaluation |
| `imbalanced-learn` | ≥ 0.9 | SMOTE oversampling |
| `joblib` | ≥ 1.0 | Model serialisation |
 
---
 
## ⚠️ Disclaimer
 
This project is for **educational and research purposes only**. It is not intended for clinical diagnosis or medical decision-making. Always consult a qualified medical professional for health concerns.
 
---
 
## License
 
This project is open-source and available under the [MIT License](LICENSE).
