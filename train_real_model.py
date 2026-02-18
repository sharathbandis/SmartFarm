import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
import io

# 1. Download the Dataset from GitHub
url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
print("Downloading dataset from GitHub...")
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))

print("Dataset Downloaded!")
print(f"Contains {len(df)} rows of real farming data.")

# 2. Prepare Data
# The CSV has columns: N, P, K, temperature, humidity, ph, rainfall, label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# 3. Train the Model
print("Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# 4. Check Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model Trained! Accuracy: {accuracy * 100:.2f}%")

# 5. Save the File
joblib.dump(model, 'RandomForest.pkl')
print("Saved as 'RandomForest.pkl'. You are ready to launch!")