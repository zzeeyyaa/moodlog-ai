import pandas as pd
import random

def generate_dummy_data(n=100):
    data = []
    for _ in range(n):
        jam_tidur = round(random.uniform(4.0, 9.0), 1)
        kualitas_tidur = random.randint(0, 3)
        mood = max(0, min(3, int(jam_tidur - 5) + random.randint(-1, 1)))
        stress = max(0, min(3, 3 - kualitas_tidur + random.randint(-1, 1)))
        aktivitas = random.choice(["kerja", "scroll hp", "jalan-jalan", "olahraga", "nugas"])
        kafein = random.choice(["Ya", "Tidak"])
        catatan = random.choice(["-", "mimpi buruk", "banyak tugas", "nyaman", "sakit kepala"])
        data.append([jam_tidur, kualitas_tidur, mood, stress, aktivitas, kafein, catatan])
    
    return pd.DataFrame(data, columns=["Jam Tidur", "Kualitas Tidur", "Mood", "Stress Level", "Aktivitas", "Kafein", "Catatan"])

df_dummy = generate_dummy_data()
df_dummy.to_csv("moodlog_dummy.csv", index=False)
