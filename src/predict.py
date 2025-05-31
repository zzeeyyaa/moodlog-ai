import joblib
import pandas as pd
import os
from datetime import datetime

# load model and encoder first ya
model = joblib.load("model/mood_model.pkl")
le_aktivitas = joblib.load("model/le_aktivitas.pkl")
le_kafein = joblib.load("model/le_kafein.pkl")

#  input manual user
print("/nMasukkan data harian kamu:")
jam_tidur = float(input("🛌 Jam Tidur (contoh: 6.5): "))
kualitas_tidur = int(input("🌙 Kualitas Tidur (0=buruk, 1=oke, 2=baik, 3=sangat baik): "))
aktivitas = input("🏃🏻‍♂️ Aktivitas utama hari ini (misal: kerja, nugas, olahraga): ").strip().lower()
kafein = input("☕ Minum kafein hari ini? (Ya/Tidak): ").strip().capitalize()

# buat input manual jadi dataframe input
input_user = {
    "Jam Tidur": jam_tidur,
    "Kualitas Tidur": kualitas_tidur,
    "Aktivitas": aktivitas,
    "Kafein": kafein
}

# ubah jadi dataframe agar bisa diproses seperti data training
df_input = pd.DataFrame([input_user])

# encode kolom kategorikal menggunakan encode hasil training
df_input["Aktivitas"] = le_aktivitas.transform(df_input['Aktivitas'])
df_input["Kafein"] = le_kafein.transform(df_input['Kafein'])

# prediksi mood disini
predicted_mood = model.predict(df_input)[0]
mood_desc = {
    0: "😔 Sangat buruk",
    1: "🙄 Agak buruk",
    2: "🙂 Netral",
    3: "😆 Baik / Semangat"
}
print(f"Prediksi mood kamu hari ini: {predicted_mood} --> {mood_desc[predicted_mood]} (skala 0-3)")

# simpan ke logs
log_path = "logs/mood_log.csv"
entry = {
    "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Jam Tidur": jam_tidur,
    "Kualitas Tidur": kualitas_tidur,
    "Aktivitas": aktivitas,
    "Kafein": kafein,
    "Mood (Prediksi)": predicted_mood,
    "Deskripsi": mood_desc[predicted_mood]
}

# append to CSV
log_df = pd.DataFrame([entry])
if os.path.exists(log_path):
    log_df.to_csv(log_path, mode='a', header=False, index=False)
else:
    log_df.to_csv(log_path, index=False)
    
print("✅ Data harian kamu berhasil disimpan ke log.")