from sklearn.ensemble import RandomForestClassifier

def build_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )