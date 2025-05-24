import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(
    page_title="K-Means Clustering - Elbow Method",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 K-Means Clustering dengan Metode Elbow Precision")
st.markdown("Optimalkan jumlah kluster menggunakan Metode Elbow yang diprecise dari sudut siku.")

with st.sidebar:
    st.header("Pengaturan Data")
    uploaded_file = st.file_uploader(
        "Upload file data (.csv atau .xlsx)",
        type=["csv", "xlsx"],
        key="file_uploader"
    )
    if uploaded_file is None:
        n = st.number_input("Jumlah data", min_value=1, value=5, step=1)
        dim = st.number_input("Jumlah dimensi", min_value=1, value=2, step=1)
    max_k = st.number_input(
        "Max jumlah kluster", min_value=2,
        value=10 if 'n' not in locals() else min(10, n), step=1
    )
    st.markdown("---")
    st.markdown("**Tips:** Upload file CSV/XLSX dengan setiap kolom sebagai dimensi, atau masukkan manual di area utama.")

# Load or input data
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            data_df = pd.read_csv(uploaded_file)
        else:
            data_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()
    n, dim = data_df.shape
    st.subheader("📂 Data dari File")
    st.dataframe(data_df, use_container_width=True)
else:
    st.subheader("📋 Input Data Manual")
    columns = [f"Dimensi {j+1}" for j in range(dim)]
    default_data = pd.DataFrame([[0.0]*dim for _ in range(n)], columns=columns)
    data_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )

# Convert to numpy
try:
    data_np = data_df.to_numpy(dtype=float)
except Exception:
    st.error("Data tidak valid: pastikan semua nilai numerik dan tidak ada cell kosong.")
    st.stop()

# ===== Preprocessing: Standardize and Winsorize outliers =====
# 1. Transformasi ke normal standar (Z-score)
means = data_np.mean(axis=0)
stds = data_np.std(axis=0)
data_scaled = (data_np - means) / stds

# 2. Cari outliers dengan IQR dan 3. Ganti dengan bounds (Winsorizing)
for j in range(data_scaled.shape[1]):
    col = data_scaled[:, j]
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # Clip outliers to bounds
    data_scaled[:, j] = np.clip(col, lower, upper)

# Gunakan data_scaled sebagai input k-means
data_processed = data_scaled.copy()

# K-means implementation
def compute_kmeans(data, k, max_iter=100):
    n_samples = data.shape[0]
    indices = random.sample(range(n_samples), k)
    centroids = data[indices].copy()
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        new_labels = np.array([np.argmin([np.linalg.norm(x - c) for c in centroids]) for x in data])
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            pts = data[labels == i]
            if pts.size:
                centroids[i] = pts.mean(axis=0)
    sse = sum(np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data)))
    return labels, centroids, sse

# Compute SSE list
sse_list = []
all_labels = []
all_centroids = []
for k in range(1, max_k + 1):
    labels, centroids, sse = compute_kmeans(data_processed.copy(), k)
    sse_list.append(sse)
    all_labels.append(labels)
    all_centroids.append(centroids)

# Precision Elbow: compute tan(psi)
tanpsi = [0]
n = len(sse_list)
for k in range(1, n-1):
    prev = sse_list[k] - sse_list[k-1]
    next_ = sse_list[k+1] - sse_list[k]
    num = -sse_list[k+1] + 2*sse_list[k] - sse_list[k-1]
    den = 1 + prev * next_
    tanpsi.append(num / den)
# Boundary
if n > 1:
    tanpsi.append(0)

# Optimal k: smallest tanpsi (most negative)
opt_k = int(np.argmin(tanpsi) + 1)

# Display results
st.markdown("---")
col1, col2 = st.columns([3, 5])
with col1:
    st.subheader("📈 Grafik SSE dan tan(ψ) vs Kluster")
    fig = go.Figure(layout=go.Layout(
        template="plotly_white",
        title=dict(text="SSE dan tan(ψ) vs Jumlah Kluster", x=0.5),
        xaxis=dict(title="Jumlah Kluster (K)", tickmode="linear", dtick=1, showgrid=False),
        yaxis=dict(title="SSE", showgrid=True, gridcolor="lightgrey")
    ))
    fig.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=sse_list, mode='lines+markers', marker=dict(size=8), line=dict(width=2), name='SSE'))
    fig.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=tanpsi, mode='lines+markers', marker_symbol='x', marker=dict(size=8), line=dict(dash='dash'), name='tan(ψ)', yaxis='y2'))
    fig.update_layout(
        yaxis2=dict(title='tan(ψ)', overlaying='y', side='right')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🔎 Kluster optimal (precision): {opt_k}")
with col2:
    st.subheader("📊 Hasil Klasterisasi")
    for i in range(opt_k):
        pts = data_processed[all_labels[opt_k-1] == i]
        count = pts.shape[0]
        with st.expander(f"Cluster {i+1} ({count} data points)"):
            if count:
                df_cluster = pd.DataFrame(pts, columns=data_df.columns)
                st.dataframe(df_cluster, use_container_width=True)
            else:
                st.write("Tidak ada data pada kluster ini.")

st.markdown("---")
st.markdown("© 2025 - Clustering KMeans | Alfin")
