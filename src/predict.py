import joblib
import pandas as pd
import streamlit as st
import os
from datetime import datetime

# load model and encoder first ya
model = joblib.load("model/mood_model.pkl")
le_bergadang = joblib.load("model/le_bergadang.pkl")
le_aktivitas = joblib.load("model/le_aktivitas.pkl")
le_kafein = joblib.load("model/le_kafein.pkl")

aktivitas_list = [
    "kerja", "scroll hp", "jalan-jalan", "olahraga", "nugas", "masak", "hobi", "kencan", "nonton", "meeting", "tiduran"
]

#  input manual user
# print("\nMasukkan data harian kamu:")
# jam_tidur = float(input("🛌 Jam Tidur (contoh: 6.5): "))
# kualitas_tidur = int(input("🌙 Kualitas Tidur (0=buruk, 1=oke, 2=baik, 3=sangat baik): "))
# bergadang = input("🕔 Bergadang hari ini? (Ya/Tidak): ").strip().capitalize()
# kafein = input("☕ Minum kafein hari ini? (Ya/Tidak): ").strip().capitalize()
# screen_time = float(input("📱 Screen Time (jam total hari ini, contoh: 4.5): "))
# frekuensi_makan = int(input("🍽️ Frekuensi makan hari ini (contoh: 3): "))
# konsumsi_air = float(input("💧 Konsumsi air (liter) hari ini (contoh: 1.5): "))
# print("\n🏃🏻‍♂️ Pilih aktivitas utama hari ini:")
# for i, akt in enumerate(aktivitas_list, 1):
#     print(f"{i}. {akt}")
# # aktivitas = input("🏃‍♂️ Aktivitas utama hari ini (misal: kerja, nugas, olahraga): ").strip().lower()
# aktivitas_idx = int(input("Masukkan nomor aktivitas (1-10): "))
# aktivitas = aktivitas_list[aktivitas_idx - 1]

# cobain streamlit
st.title("📊 Mood Predictor Harian")

with st.form("form_mood"):
    jam_tidur = st.slider("🛌 Jam Tidur", 4.0, 9.0, 6.5, 0.5)
    kualitas_tidur = st.selectbox("🌙 Kualitas Tidur", [0, 1, 2, 3])
    bergadang = st.radio("🕔 Bergadang hari ini?", ["Ya", "Tidak"])
    kafein = st.radio("☕ Minum kafein hari ini?", ["Ya", "Tidak"])
    screen_time = st.slider("📱 Screen Time (jam)", 0.0, 12.0, 4.0, 0.5)
    frekuensi_makan = st.slider("🍽️ Frekuensi Makan", 1, 5, 3)
    konsumsi_air = st.slider("💧 Konsumsi Air (liter)", 0.5, 3.5, 1.5, 0.1)
    aktivitas = st.selectbox("🏃🏻‍♂️ Aktivitas Utama", aktivitas_list)
    
    submitted = st.form_submit_button("Prediksi Mood")


if submitted:
    # Buat dataframe input
    df_input = pd.DataFrame([{
        "Jam Tidur": jam_tidur,
        "Kualitas Tidur": kualitas_tidur,
        "Bergadang": le_bergadang.transform([bergadang])[0],
        "Kafein": le_kafein.transform([kafein])[0],
        "Screen Time": screen_time,
        "Frekuensi Makan": frekuensi_makan,
        "Konsumsi Air": konsumsi_air,
        "Aktivitas": le_aktivitas.transform([aktivitas])[0],
    }])

    # Susun urutan kolom agar sesuai model
    df_input = df_input[[
        "Jam Tidur", "Kualitas Tidur", "Bergadang", "Screen Time",
        "Aktivitas", "Kafein", "Frekuensi Makan", "Konsumsi Air"
    ]]

    # Prediksi dan tampilkan hasil
    pred = round(model.predict(df_input)[0])
    mood_desc = {
        0: "😔 Sangat buruk",
        1: "🙄 Agak buruk",
        2: "🙂 Netral",
        3: "😊 Cukup baik",
        4: "😆 Sangat baik / Semangat"
    }
    st.success(f"Prediksi Mood Kamu: {pred} → {mood_desc[pred]} (skala 0–4)")

    # Simpan ke log
    log_path = "logs/mood_log.csv"
    log_entry = {
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Jam Tidur": jam_tidur,
        "Kualitas Tidur": kualitas_tidur,
        "Bergadang": bergadang,
        "Screen Time": screen_time,
        "Aktivitas": aktivitas,
        "Kafein": kafein,
        "Frekuensi Makan": frekuensi_makan,
        "Konsumsi Air": konsumsi_air,
        "Mood (Prediksi)": pred,
        "Deskripsi": mood_desc[pred]
    }
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    st.info("✅ Data harian kamu berhasil disimpan ke log.")
    
# buat input manual jadi dataframe input
# input_user = {
#     "Jam Tidur": jam_tidur,
#     "Kualitas Tidur": kualitas_tidur,
#     "Bergadang":bergadang,
#     "Kafein": kafein,
#     "Screen Time": screen_time,
#     "Frekuensi Makan": frekuensi_makan,
#     "Konsumsi Air": konsumsi_air,
#     "Aktivitas": aktivitas
# }

# ubah jadi dataframe agar bisa diproses seperti data training
# df_input = pd.DataFrame([input_user])

# # encode kolom kategorikal menggunakan encode hasil training
# df_input["Bergadang"] = le_bergadang.transform(df_input['Bergadang'])
# df_input["Kafein"] = le_kafein.transform(df_input['Kafein'])
# df_input["Aktivitas"] = le_aktivitas.transform(df_input['Aktivitas'])

# # Urutkan kolom sesuai model training
# df_input = df_input[[
#     "Jam Tidur", "Kualitas Tidur", "Bergadang", "Screen Time",
#     "Aktivitas", "Kafein", "Frekuensi Makan", "Konsumsi Air"
# ]]

# # prediksi mood disini
# # predicted_mood = model.predict(df_input)[0]
# predicted_mood = round(model.predict(df_input)[0])

# mood_desc = {
#     0: "😔 Sangat buruk",
#     1: "🙄 Agak buruk",
#     2: "🙂 Netral",
#     3: "😊 Cukup baik",
#     4: "😆 Sangat baik / Semangat"
# }
# print(f"Prediksi mood kamu hari ini: {predicted_mood} --> {mood_desc[predicted_mood]} (skala 0-4)")

# # simpan ke logs
# log_path = "logs/mood_log.csv"
# entry = {
#     "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "Jam Tidur": jam_tidur,
#     "Kualitas Tidur": kualitas_tidur,
#     "Bergadang": bergadang,
#     "Screen Time": screen_time,
#     "Aktivitas": aktivitas,
#     "Kafein": kafein,
#     "Frekuensi Makan": frekuensi_makan,
#     "Konsumsi Air": konsumsi_air,
#     "Mood (Prediksi)": predicted_mood,
#     "Deskripsi": mood_desc[predicted_mood]
# }

# # append to CSV
# log_df = pd.DataFrame([entry])
# if os.path.exists(log_path):
#     log_df.to_csv(log_path, mode='a', header=False, index=False)
# else:
#     log_df.to_csv(log_path, index=False)
    
# print("✅ Data harian kamu berhasil disimpan ke log.")