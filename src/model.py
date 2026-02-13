from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_models():
    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]),
        "decision_tree": DecisionTreeClassifier()
    }
    return models
