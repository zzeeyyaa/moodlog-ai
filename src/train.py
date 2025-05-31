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
    
    # split data to data train and data test
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
    
    # build and train model
    model = fine_tune_model(X_train, y_train, X_test, y_test)
    # model.fit(X_train, y_train)
    
    # evaluate
    y_pred = model.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    # print(f"Report klasifikasi: {report}")
    # print(f"Akurasi model: {acc:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # save model and encoders
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/mood_model.pkl")
    joblib.dump(le_bergadang, "model/le_bergadang.pkl")
    joblib.dump(le_aktivitas, "model/le_aktivitas.pkl")
    joblib.dump(le_kafein, "model/le_kafein.pkl")
    print("Model dan encoder disimpan di folder `model/`")
    

if __name__ == '__main__':
    train_and_save_model()