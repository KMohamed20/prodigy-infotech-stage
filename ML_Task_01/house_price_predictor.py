"""
Machine Learning Task 01 - House Price Prediction
Prodigy InfoTech Internship | Khalid Ag Mohamed Aly
August 2025

Objective: Predict house prices using Random Forest based on realistic synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'surface': np.random.uniform(40, 300, n_samples),
    'chambres': np.random.randint(1, 7, n_samples),
    'salle_bain': np.random.randint(1, 5, n_samples),
    'age': np.random.uniform(0, 50, n_samples),
    'garage': np.random.choice([0, 1], n_samples),
    'quartier_score': np.random.uniform(1, 10, n_samples)
}

# Realistic price formula with noise
prix_base = (
    data['surface'] * np.random.normal(2000, 200) +
    data['chambres'] * np.random.normal(15000, 2000) +
    data['salle_bain'] * np.random.normal(10000, 1500) -
    data['age'] * np.random.normal(1000, 200) +
    data['garage'] * np.random.normal(20000, 3000) +
    data['quartier_score'] * np.random.normal(5000, 1000)
)
prix = prix_base + np.random.normal(0, 15000, n_samples)

data['prix'] = np.clip(prix, 50000, 800000)

df = pd.DataFrame(data)

# 2. Preprocessing
X = df.drop('prix', axis=1)
y = df['prix']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model training
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:,.0f}€")
print(f"R²: {r2:.3f}")
print(f"MAE: {mae:,.0f}€")

# 5. Feature importance
feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# 6. Save model and scaler
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 7. Prediction function
def predict_price(surface, chambres, salle_bain, age, garage, quartier_score):
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    input_data = np.array([[surface, chambres, salle_bain, age, garage, quartier_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)

# Example usage
example_price = predict_price(150, 4, 2, 10, 1, 8.5)
print(f"\nPrix prédit pour maison (150m², 4 chambres, etc.) : {example_price:,.0f}€")
