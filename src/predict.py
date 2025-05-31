import joblib
import pandas as pd
import os
from datetime import datetime

# load model and encoder first ya
model = joblib.load("model/mood_model.pkl")
le_bergadang = joblib.load("model/le_bergadang.pkl")
le_aktivitas = joblib.load("model/le_aktivitas.pkl")
le_kafein = joblib.load("model/le_kafein.pkl")

#  input manual user
print("\nMasukkan data harian kamu:")
jam_tidur = float(input("🛌 Jam Tidur (contoh: 6.5): "))
kualitas_tidur = int(input("🌙 Kualitas Tidur (0=buruk, 1=oke, 2=baik, 3=sangat baik): "))
bergadang = input("🕔 Bergadang hari ini? (Ya/Tidak): ").strip().capitalize()
kafein = input("☕ Minum kafein hari ini? (Ya/Tidak): ").strip().capitalize()
screen_time = float(input("📱 Screen Time (jam total hari ini, contoh: 4.5): "))
frekuensi_makan = int(input("🍽️ Frekuensi makan hari ini (contoh: 3): "))
konsumsi_air = float(input("💧 Konsumsi air (liter) hari ini (contoh: 1.5): "))
aktivitas = input("🏃‍♂️ Aktivitas utama hari ini (misal: kerja, nugas, olahraga): ").strip().lower()


# buat input manual jadi dataframe input
input_user = {
    "Jam Tidur": jam_tidur,
    "Kualitas Tidur": kualitas_tidur,
    "Bergadang":bergadang,
    "Kafein": kafein,
    "Screen Time": screen_time,
    "Frekuensi Makan": frekuensi_makan,
    "Konsumsi Air": konsumsi_air,
    "Aktivitas": aktivitas
}

# ubah jadi dataframe agar bisa diproses seperti data training
df_input = pd.DataFrame([input_user])

# encode kolom kategorikal menggunakan encode hasil training
df_input["Bergadang"] = le_bergadang.transform(df_input['Bergadang'])
df_input["Kafein"] = le_kafein.transform(df_input['Kafein'])
df_input["Aktivitas"] = le_aktivitas.transform(df_input['Aktivitas'])

# Urutkan kolom sesuai model training
df_input = df_input[[
    "Jam Tidur", "Kualitas Tidur", "Bergadang", "Screen Time",
    "Aktivitas", "Kafein", "Frekuensi Makan", "Konsumsi Air"
]]

# prediksi mood disini
# predicted_mood = model.predict(df_input)[0]
predicted_mood = round(model.predict(df_input)[0])

mood_desc = {
    0: "😔 Sangat buruk",
    1: "🙄 Agak buruk",
    2: "🙂 Netral",
    3: "😊 Cukup baik",
    4: "😆 Sangat baik / Semangat"
}
print(f"Prediksi mood kamu hari ini: {predicted_mood} --> {mood_desc[predicted_mood]} (skala 0-4)")

# simpan ke logs
log_path = "logs/mood_log.csv"
entry = {
    "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Jam Tidur": jam_tidur,
    "Kualitas Tidur": kualitas_tidur,
    "Bergadang": bergadang,
    "Screen Time": screen_time,
    "Aktivitas": aktivitas,
    "Kafein": kafein,
    "Frekuensi Makan": frekuensi_makan,
    "Konsumsi Air": konsumsi_air,
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