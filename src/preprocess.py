import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given path.
    """
    data = pd.read_csv(path)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning:
    - Remove duplicates
    - Drop missing values
    """
    data = data.drop_duplicates()
    data = data.dropna()
    return data


def split_features_target(data: pd.DataFrame):
    """
    Separate features and target variable.
    """
    X = data[['study_hours', 'attendance', 'previous_marks']]
    y = data['result']
    return X, y


def preprocess_pipeline(path: str):
    """
    Complete preprocessing workflow.
    """
    data = load_data(path)
    data = clean_data(data)
    X, y = split_features_target(data)
    return X, y
