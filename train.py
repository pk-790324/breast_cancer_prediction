import pandas as pd
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load Dataset
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("Original dataset shape:", X.shape)


# ==========================================
# 2. Select 5 Features
# ==========================================

selected_features = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean compactness",
    "mean concavity"
]

X = X[selected_features]

print("Selected features:")
for feature in selected_features:
    print("-", feature)


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Preprocessing
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. Train Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ==========================================
# 6. Evaluate
# ==========================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X.shape[1])
print("Accuracy:", accuracy)


# ==========================================
# 7. Save Model
# ==========================================

model_data = {
    "model": model,
    "scaler": scaler,
    "feature_names": selected_features,
    "target_names": list(data.target_names)
}

with open("breast_cancer_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved as breast_cancer_model.pkl")