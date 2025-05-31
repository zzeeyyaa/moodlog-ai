import pandas as pd
import numpy as np
import random

def generate_dummy_data_balanced(per_class, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    buckets = {i: [] for i in range(5)}  # Mood: 0–4

    while not all(len(bucket) >= per_class for bucket in buckets.values()):
        # Basic features
        jam_tidur = round(random.uniform(4.0, 9.0), 1)
        kualitas_tidur = random.randint(0, 5)
        screen_time = round(random.uniform(0.5, 10.0), 1)
        konsumsi_air = round(random.uniform(0.5, 3.5), 1)
        makan_perhari = random.randint(1, 5)
        bergadang = random.choice(["Ya", "Tidak"])
        kafein = random.choice(["Ya", "Tidak"])
        aktivitas = random.choice(["kerja", "scroll hp", "jalan-jalan", "olahraga", "nugas", "masak", "hobi", "dating", "kencan"])
        catatan = random.choice(["-", "mimpi buruk", "banyak tugas", "nyaman", "sakit kepala", "kena sial", "ketiban sial", "diputusin", "galau", "sakit hati",
                                 "jadian", "happy", "senang", "bahagia", "dapat uang", "gajian",
                                 "sakit gigi", "sakit perut", "mual", "sial", "mendapatkan gaji", "dapat gaji"])

        # Mood score base
        mood_score = (
            (jam_tidur * 0.8) +
            (kualitas_tidur * 0.5) -
            (screen_time * 0.3)
        )

        # Stress score base
        stress_score = (5 - kualitas_tidur) * 0.7

        # Pengaruh screen time
        if screen_time > 6:
            mood_score -= 1
            stress_score += 1
        elif screen_time < 3:
            mood_score += 1
            stress_score -= 1

        # Bergadang
        if bergadang == "Ya" and jam_tidur < 6:
            mood_score -= 1
            stress_score += 1

        # Kafein
        if kafein == "Ya":
            mood_score += random.choice([0, 1])
            if kualitas_tidur < 3:
                stress_score += 1

        # Pola makan
        if makan_perhari <= 2:
            mood_score -= 1
            stress_score += 1
        elif makan_perhari >= 5:
            stress_score += 1

        # Konsumsi air
        if konsumsi_air < 1.0:
            mood_score -= 1
            stress_score += 1
        elif konsumsi_air >= 2.0:
            mood_score += 1
            stress_score -= 1

        # Aktivitas
        if aktivitas in ["olahraga", "jalan-jalan", "hobi", "dating", "kencan"]:
            mood_score += 1
            stress_score -= 1
        elif aktivitas in ["scroll hp", "nugas"]:
            mood_score -= 1
            stress_score += 1
        elif aktivitas == "masak":
            mood_score += 1
        elif aktivitas == "kerja":
            mood_score += random.choice([0, 1])
            stress_score += random.choice([0, 1])

        # Catatan
        if catatan in ["mimpi buruk", "kena sial", "sial", "ketiban sial", "diputusin", "galau", "sakit hati"]:
            mood_score -= 1
            stress_score += 1
        elif catatan == "banyak tugas":
            stress_score += 1
        elif catatan in ["nyaman", "jadian", "happy", "senang", "bahagia", "dapat uang", "gajian", "mendapatkan gaji", "dapat gaji"]:
            mood_score += 1
            stress_score -= 1
        elif catatan in ["sakit kepala", "sakit gigi", "sakit perut", "mual"]:
            mood_score -= 1
            stress_score += 1

        # Final mood dan stress
        mood = int(np.clip(round(mood_score), 0, 4))
        stress = int(np.clip(round(stress_score), 0, 4))

        if len(buckets[mood]) < per_class:
            buckets[mood].append([
                jam_tidur, kualitas_tidur, bergadang, kafein, makan_perhari, konsumsi_air,
                screen_time, aktivitas, catatan, mood, stress
            ])

    # Gabungkan semua bucket
    data = sum(buckets.values(), [])
    df = pd.DataFrame(data, columns=[
        "Jam Tidur", "Kualitas Tidur", "Bergadang", "Kafein", "Frekuensi Makan", "Konsumsi Air",
        "Screen Time", "Aktivitas", "Catatan", "Mood", "Stress Level"
    ])
    
    return df


# Generate dan simpan
df_dummy = generate_dummy_data_balanced(per_class=400)
df_dummy.to_csv("data/moodlog_dummy.csv", index=False)

# Optional: tampilkan distribusi
print("Distribusi Mood:\n", df_dummy["Mood"].value_counts().sort_index())
print("Distribusi Stress:\n", df_dummy["Stress Level"].value_counts().sort_index())
