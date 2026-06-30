import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model directory
os.makedirs('model', exist_ok=True)

print("Loading dataset...")
# Load dataset from the Rain-Prediction repository folder
df = pd.read_csv('../Rain-Prediction/weatherAUS.csv')

cols = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 
    'RainToday', 'RainTomorrow'
]

# Drop missing values in our features and target
data = df[cols].dropna()

# Map RainToday and RainTomorrow to binary 0/1
data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

print(f"Training set shape: {X.shape}")

# Train Random Forest Classifier
print("Training RandomForest model...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X, y)

# Save model and feature list
joblib.dump(rfc, 'model/rf_rain_model.pkl')
joblib.dump(list(X.columns), 'model/features.pkl')

print("Model trained and saved successfully!")
