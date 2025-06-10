from src.data_loader import load_and_preprocess
from src.model import fine_tune_model
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error

import joblib
import os

def train_and_save_model():
    # tryna load data and encode
    X, y,le_bergadang, le_aktivitas, le_kafein = load_and_preprocess()
    
    # split data to data train and data temp
    X_train, X_temp, y_train, y_temp =  train_test_split(X, y, test_size=0.4, random_state=42)
    
    # split data to data test and data cv
    X_test, X_cv, y_test, y_cv = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # build and train model using data train
    model = fine_tune_model(X_train, y_train)
    
    # Evaluation using cross validation
    y_pred_cv = model.predict(X_cv)
    mse_cv = mean_squared_error(y_cv, y_pred_cv)
    mae_cv = mean_absolute_error(y_cv, y_pred_cv)
    print(f"Cross Validation MSE: {mse_cv: .2f}")
    print(f"Cross Validation MAE: {mae_cv: .2f}")
    
    # Final evaluation using data test
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"Test MSE: {mse_test: .2f}")
    print(f"Test MAE: {mae_test: .2f}")
    
    # save model and encoders
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/mood_model.pkl")
    joblib.dump(le_bergadang, "model/le_bergadang.pkl")
    joblib.dump(le_aktivitas, "model/le_aktivitas.pkl")
    joblib.dump(le_kafein, "model/le_kafein.pkl")
    print("Model dan encoder disimpan di folder `model/`")
    

if __name__ == '__main__':
    train_and_save_model()