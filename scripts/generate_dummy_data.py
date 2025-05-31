import pandas as pd
import random

def generate_dummy_data_balanced(per_class=100):
    buckets = {i: [] for i in range(6)}  # mood 0–5
    
    while not all(len(bucket) >= per_class for bucket in buckets.values()):
        jam_tidur = round(random.uniform(4.0, 9.0), 1)
        kualitas_tidur = random.randint(0, 5)
        
        # Hitung mood awal
        mood = max(0, min(5, int((jam_tidur - 4) * 1.2) + random.randint(-1, 1)))
        stress = max(0, min(5, int(5 - kualitas_tidur + random.randint(-1, 1))))
        
        bergadang = random.choice(["Ya", "Tidak"])
        if bergadang == "Ya" and jam_tidur < 6:
            mood -= 1
            stress += 1

        kafein = random.choice(["Ya", "Tidak"])
        if kafein == "Ya":
            mood += random.choice([0, 1])
        if kafein == "Ya" and kualitas_tidur < 3:
            stress += 1

        aktivitas = random.choice(["kerja", "scroll hp", "jalan-jalan", "olahraga", "nugas", "masak"])
        catatan = random.choice(["-", "mimpi buruk", "banyak tugas", "nyaman", "sakit kepala"])
        
        # Efek aktivitas
        if aktivitas == "olahraga":
            mood += 1
            stress -= 1
        elif aktivitas == "jalan-jalan":
            mood += 1
            stress -= 1
        elif aktivitas == "scroll hp":
            mood -= 1
            stress += 1
        elif aktivitas == "nugas":
            mood -= 1
            stress += 1
        elif aktivitas == "masak":
            mood += 1
        elif aktivitas == "kerja":
            stress += random.choice([0, 1])
            mood += random.choice([0, 1])

        # Batas bawah dan atas
        mood = max(0, min(5, mood))
        stress = max(0, min(5, stress))

        # Simpan hanya kalau bucket mood belum penuh
        if len(buckets[mood]) < per_class:
            buckets[mood].append([
                jam_tidur, kualitas_tidur, bergadang,
                mood, stress, aktivitas, kafein, catatan
            ])

    # Gabungkan semua bucket
    data = sum(buckets.values(), [])
    df = pd.DataFrame(data, columns=[
        "Jam Tidur", "Kualitas Tidur", "Bergadang",
        "Mood", "Stress Level", "Aktivitas", "Kafein", "Catatan"
    ])
    
    return df

# Generate dan simpan
df_dummy = generate_dummy_data_balanced(per_class=100)
df_dummy.to_csv("data/moodlog_dummy.csv", index=False)

# Optional: tampilkan distribusi
print(df_dummy["Mood"].value_counts().sort_index())
