from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

def fine_tune_model(X_train, y_train):
    # Kolom numerik dan teks
    num_cols = ['Jam Tidur', 'Kualitas Tidur', 'Bergadang', 'Screen Time', 'Aktivitas', 'Kafein', 'Frekuensi Makan', 'Konsumsi Air']
    text_col = 'Catatan'
    
    # Pipeline preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('txt', TfidfVectorizer(), text_col)
        ]
    )

    # Pipeline model
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])

    # Parameter grid XGBoost
    param_grid = {
        'regressor__n_estimators': [100, 300, 500],
        'regressor__max_depth': [3, 6, 10],
        'regressor__learning_rate': [0.01, 0.1, 0.3],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               cv=3, scoring='neg_mean_absolute_error',
                               verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    return best_model
