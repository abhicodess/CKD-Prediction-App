import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Keep only relevant columns
df = df[['age', 'bp', 'rbc', 'pc', 'pcc', 'ba', 'classification']]

# Drop rows with missing values in key features or target
df.dropna(subset=['age', 'bp', 'rbc', 'pc', 'pcc', 'ba', 'classification'], inplace=True)

# Encode features
df['rbc'] = df['rbc'].map({'normal': 1, 'abnormal': 0})
df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
df['pcc'] = df['pcc'].map({'present': 1, 'notpresent': 0})
df['ba'] = df['ba'].map({'present': 1, 'notpresent': 0})

# Clean and encode target
df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})
df = df.dropna(subset=['classification'])  # Drop rows where mapping failed

# Split into input and target
X = df[['age', 'bp', 'rbc', 'pc', 'pcc', 'ba']]
y = df['classification']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("CKD.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as CKD.pkl")
