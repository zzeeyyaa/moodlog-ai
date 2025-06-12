from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np

def fine_tune_model(X_train, y_train):
    # Daftar kolom numerik/kategori dan kolom teks
    num_cols = ['Jam Tidur', 'Kualitas Tidur', 'Bergadang', 'Screen Time', 'Aktivitas', 'Kafein', 'Frekuensi Makan', 'Konsumsi Air']
    text_col = 'Catatan'
    
    # Pipeline preprocessing: gabung numerik dengna text data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('txt', TfidfVectorizer(), text_col)
        ]
    )
    # Pipeline model
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # gridsearch param pipeline
    # param_grid = {
    #     'n_estimators': [200, 500, 800],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2']
    # }

    param_grid = {
    'regressor__n_estimators': [200, 500, 800],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               cv=3, scoring='neg_mean_absolute_error',
                               verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    return best_model
# 
    # Fitting 3 folds for each of 162 candidates, totalling 486 fits
    # Best Parameters: {'regressor__max_depth': None, 'regressor__max_features': 'sqrt', 'regressor__min_samples_leaf': 1, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 800}
    # Cross Validation MSE:  0.81
    # Cross Validation MAE:  0.75
    # Test MSE:  0.71
    # Test MAE:  0.72
    # Cross Validation MAE:  0.75
    # Test MSE:  0.71
    # Test MAE:  0.72
    # Model dan encoder disimpan di folder `model/`
# 