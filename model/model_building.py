import numpy as np
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# 2. Select EXACTLY six approved features
selected_features = [
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'total_phenols',
    'proline'
]

feature_indices = [feature_names.index(f) for f in selected_features]
X = X[:, feature_indices]

# 3. Handle missing values (dataset has none, but included for compliance)
X = np.nan_to_num(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature scaling (MANDATORY)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model (KNN – allowed algorithm)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Cultivar 1", "Cultivar 2", "Cultivar 3"]
))

# 8. Save model + scaler together (Pickle)
with open("wine_cultivar_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": selected_features
        },
        f
    )

print("\n✅ Wine cultivar model saved successfully")


