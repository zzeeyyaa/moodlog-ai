import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np

def fine_tune_model(X_train, y_train, X_test, y_test, n_trials=50):
    def objective(trial):
        rf = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            random_state=42,
            n_jobs=-1
        )
        score = cross_val_score(rf, X_train, y_train, cv=3, scoring="neg_mean_absolute_error")
        return -np.mean(score)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Parameters:", study.best_params)

    # Training ulang model terbaik
    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Evaluasi
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE dari model terbaik: {mae:.2f}")

    return best_model
