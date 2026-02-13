import sys
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocess import load_data, split_features_target
from src.model import get_models

data = load_data("data/students.csv")
X, y = split_features_target(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = get_models()

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model

joblib.dump(best_model, "models/pipeline.pkl")

print(f"\nBest Model Accuracy: {best_score}")
print("Pipeline saved successfully!")
