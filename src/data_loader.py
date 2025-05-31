import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath="data/moodlog_dummy.csv"):
    df = pd.read_csv(filepath)
    
    X = df[["Jam Tidur", "Kualitas Tidur", "Bergadang", "Aktivitas", "Kafein"]].copy()
    y = df["Mood"]
    
    le_bergadang = LabelEncoder()
    le_aktivitas = LabelEncoder()
    le_kafein = LabelEncoder()
    
    X["Bergadang"] = le_bergadang.fit_transform(X["Bergadang"])
    X["Aktivitas"] = le_aktivitas.fit_transform(X["Aktivitas"])
    X["Kafein"] = le_kafein.fit_transform(X["Kafein"])
    
    return X, y, le_bergadang, le_aktivitas, le_kafein
    
    