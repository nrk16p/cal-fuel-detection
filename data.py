import pandas as pd

# ======================================================
# 🧭 1️⃣ เตรียมข้อมูลพื้นฐาน
# ======================================================
df = df_all[['ทะเบียนพาหนะ', 'รหัสพาหนะ', 'วันที่', 'เวลา', 'วันที่สิ้นสุด',
             'เวลาสิ้นสุด', 'พิกัด', 'ความเร็ว(กม./ชม.)', 'ระยะทาง(กม.)', 'น้ำมัน', 'สถานะ']]

# 🕒 รวมวันที่และเวลาเป็นคอลัมน์ datetime เดียว
df["datetime"] = pd.to_datetime(
    df["วันที่"].astype(str) + " " + df["เวลา"].astype(str),
    dayfirst=True,  # วันที่อยู่หน้าเดือน
    errors="coerce"  # ถ้าแปลงไม่ได้ ให้เป็น NaT
)

# ======================================================
# 📊 2️⃣ รวมข้อมูลเป็นช่วงเวลา 5 นาที (Resample 5min)
# ======================================================
df = df.set_index('datetime')

df_result = df.resample('5min').agg({
    'ความเร็ว(กม./ชม.)': 'mean',  # ค่าเฉลี่ยความเร็ว
    'ระยะทาง(กม.)': 'sum',        # ผลรวมระยะทางในแต่ละช่วง
    'น้ำมัน': 'mean'               # ค่าเฉลี่ยระดับน้ำมัน
}).reset_index()

# เปลี่ยนชื่อคอลัมน์ให้อ่านง่าย
df_result.columns = ['datetime5mins', 'ความเร็ว(กม./ชม.)avg', 'ระยะทาง(กม.)avg', 'ระดับน้ำมันavg']

# เติมค่า NaN ด้วยค่าก่อนหน้า (forward fill)
df_result = df_result.ffill()

# ======================================================
# 🔁 3️⃣ ฟังก์ชันคำนวณความต่างย้อนหลัง/ล่วงหน้า
# ======================================================
def diff_backward(df, col, minutes):
    """คำนวณค่าความต่างย้อนหลัง (ย้อนหลัง n นาที)"""
    left = df[['datetime5mins', col]].copy()
    left['lookup_time'] = left['datetime5mins'] - pd.Timedelta(minutes=minutes)

    right = df[['datetime5mins', col]].rename(
        columns={'datetime5mins': 'dt_ref', col: f'{col}_ref'}
    ).sort_values('dt_ref')

    merged = pd.merge_asof(
        left.sort_values('lookup_time'),
        right, left_on='lookup_time', right_on='dt_ref',
        direction='backward', allow_exact_matches=True
    )

    return left[col].values - merged[f'{col}_ref'].values


def diff_forward(df, col, minutes):
    """คำนวณค่าความต่างล่วงหน้า (ข้างหน้า n นาที)"""
    left = df[['datetime5mins', col]].copy()
    left['lookup_time'] = left['datetime5mins'] + pd.Timedelta(minutes=minutes)

    right = df[['datetime5mins', col]].rename(
        columns={'datetime5mins': 'dt_ref', col: f'{col}_ref'}
    ).sort_values('dt_ref')

    merged = pd.merge_asof(
        left.sort_values('lookup_time'),
        right, left_on='lookup_time', right_on='dt_ref',
        direction='forward', allow_exact_matches=True
    )

    return merged[f'{col}_ref'].values - left[col].values


# ======================================================
# ⛽ 4️⃣ คำนวณค่า diff ของระดับน้ำมัน
# ======================================================
col = 'ระดับน้ำมันavg'

# ย้อนหลัง (ago)
df_result['fuel_diff_5min_ago'] = diff_backward(df_result, col, 5)
df_result['fuel_diff_60min_ago'] = diff_backward(df_result, col, 60)

# ล่วงหน้า (next)
df_result['fuel_diff_next_5min'] = diff_forward(df_result, col, 5)
df_result['fuel_diff_next_60min'] = diff_forward(df_result, col, 60)

# เติมค่า NaN ด้วยศูนย์ (กรณีต้น/ท้ายไม่มีข้อมูล)
df_result = df_result.fillna(0)

# ======================================================
# 🚨 5️⃣ สร้างคอลัมน์แจ้งเตือนเหตุการณ์
# ======================================================

# กรณีระดับน้ำมันใน 60 นาทีถัดไป "ลดลง" จากค่าเฉลี่ยปัจจุบัน
df_result['fuel_next_60mins_less_than_avg'] = df_result.apply(
    lambda row: "yes" if row['fuel_diff_next_60min'] < 0 else "no",
    axis=1
)

# กรณีระดับน้ำมันลดลงเกิน 5 ลิตรใน 5 นาที (อาจเป็นจุดเติม/ดูดน้ำมัน)
df_result['decrease_5litres'] = df_result.apply(
    lambda row: "yes" if row['fuel_diff_5min_ago'] < -5 else "no",
    axis=1
)

# ======================================================
# ✅ 6️⃣ ตรวจสอบผลลัพธ์
# ======================================================
print(df_result.head())
print(f"ข้อมูลทั้งหมด {len(df_result)} แถว")
#ผลการทดลอง plot x = 5minago and y = r=fuel rate 
#ข้อมูลระดับน้ำัมะนยังไม่มีนัยยะ 
import pandas as pd

# สมมติว่าคอลัมน์ datetime เป็น datetime แล้ว
start_time = pd.Timestamp("2025-09-13 00:00:00")
end_time   = pd.Timestamp("2025-09-13 23:59:59")

df_filtered = df_result[
    (df_result['datetime5mins'] >= start_time) &
    (df_result['datetime5mins'] <= end_time)
].copy()

print(f"ข้อมูลที่เลือก: {len(df_filtered)} แถว")
df_filtered


import matplotlib.pyplot as plt
from matplotlib import rcParams

# ตั้งค่าฟอนต์ให้เป็น Tahoma
rcParams['font.family'] = 'Tahoma'

# 🔹 พล็อต Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    df_filtered['fuel_diff_5min_ago'],
    df_filtered['fuel_diff_next_5min'],
    c='tab:blue', alpha=0.7, edgecolors='k'
)

plt.title("Scatter: Fuel Diff (5 นาที ก่อนหน้า) vs fuel_diff_next_5min	", fontsize=14)
plt.xlabel("ส่วนต่างระดับน้ำมันย้อนหลัง 5 นาที (ลิตร)", fontsize=12)
plt.ylabel("fuel_diff_next_5min	", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.stats import norm

# ตั้งค่าฟอนต์เป็น Tahoma
rcParams['font.family'] = 'Tahoma'

# เตรียมข้อมูล
data = df_filtered['fuel_diff_5min_ago'].dropna()

# คำนวณค่า mean และ std
mean = data.mean()
std = data.std()

# สร้างค่าแกน X สำหรับเส้นโค้ง normal distribution
x = np.linspace(data.min(), data.max(), 200)
pdf = norm.pdf(x, mean, std)

# 🔹 พล็อต histogram + เส้น normal distribution
plt.figure(figsize=(10,6))
plt.hist(data, bins=30, density=True, alpha=0.6, color='tab:blue', edgecolor='black', label='Histogram')
plt.plot(x, pdf, color='red', linewidth=2, label=f'Normal(μ={mean:.2f}, σ={std:.2f})')

plt.title("การกระจายตัวของ Fuel Diff (ย้อนหลัง 5 นาที)", fontsize=14)
plt.xlabel("Fuel Diff 5 นาที ก่อนหน้า (ลิตร)", fontsize=12)
plt.ylabel("ความหนาแน่น (Density)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# 🧩 ตั้งค่าฟอนต์
rcParams['font.family'] = 'Tahoma'

# 🔹 เตรียมข้อมูล
X = df_filtered[['fuel_diff_5min_ago', 'fuel_diff_next_5min']].dropna().values

# 🔹 มาตรฐานข้อมูล (สำคัญมากสำหรับ DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 สร้างโมเดล DBSCAN
# ปรับ eps / min_samples ได้ตาม density ของข้อมูล
db = DBSCAN(eps=0.5, min_samples=3).fit(X_scaled)

# 🔹 ดึง labels ออกมา
labels = db.labels_

# 🔹 แปลงผลกลับไปใส่ใน df
df_clustered = df_filtered.dropna(subset=['fuel_diff_5min_ago', 'fuel_diff_next_5min']).copy()
df_clustered['cluster'] = labels

# 🔹 นับจำนวนคลัสเตอร์
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"✅ พบ {n_clusters} คลัสเตอร์ | จุด Noise: {n_noise}")

# 🔹 พล็อตกราฟ
plt.figure(figsize=(10, 6))

# สร้าง colormap ตาม cluster
unique_labels = set(labels)
colors = plt.cm.get_cmap('tab10', len(unique_labels))

for k in unique_labels:
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    if k == -1:
        # noise (outliers)
        plt.scatter(xy[:, 0], xy[:, 1], c='gray', marker='x', label='Noise')
    else:
        plt.scatter(xy[:, 0], xy[:, 1], color=colors(k), alpha=0.7, edgecolors='k', label=f'Cluster {k}')

plt.title("DBSCAN Clustering: fuel_diff_5min_ago vs fuel_diff_next_5min", fontsize=14)
plt.xlabel("fuel_diff_5min_ago")
plt.ylabel("fuel_diff_next_5min")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# เงื่อนไขที่ 1: fuel_diff_next_60min > 0
df_clustered = df_clustered[df_clustered['fuel_diff_next_60min'] < 0]
df_clustered = df_clustered[df_clustered['fuel_diff_60min_ago'] < 0]


# เงื่อนไขที่ 2: ส่วนต่างของ diff ย้อนหลัง 5min และ 60min น้อยกว่า 10
df_clustered = df_clustered[(df_clustered['fuel_diff_5min_ago'] - df_clustered['fuel_diff_60min_ago']).abs() < 5]
df_clustered = df_clustered[(df_clustered['fuel_diff_next_5min'] - df_clustered['fuel_diff_next_60min']).abs() < 5]