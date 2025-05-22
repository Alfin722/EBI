import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import io

st.set_page_config(
    page_title="K-Means Clustering - Elbow Method",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 K-Means Clustering dengan Metode Elbow")
st.markdown("Optimalkan jumlah kluster menggunakan Metode Elbow.")

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
    max_k = st.number_input("Max jumlah kluster", min_value=2, value=10 if 'n' not in locals() else min(10, n), step=1)
    
    threshold = st.sidebar.slider("Ambang batas penurunan SSE (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    st.markdown("---")
    st.markdown("**Tips:** Upload file CSV/XLSX dengan setiap kolom sebagai dimensi, atau masukkan manual di area utama.")

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            data_df = pd.read_csv(uploaded_file)
        else:
            data_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()
    # Derive n and dim from file
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

try:
    data_np = data_df.to_numpy(dtype=float)
except Exception:
    st.error("Data tidak valid: pastikan semua nilai numerik dan tidak ada cell kosong.")
    st.stop()

def compute_kmeans(data_np, k, max_iter=100):
    n_samples, _ = data_np.shape
    indices = random.sample(range(n_samples), k)
    centroids = data_np[indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        new_labels = np.array([np.argmin([np.linalg.norm(x - c) for c in centroids]) for x in data_np])
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for i in range(k):
            pts = data_np[labels == i]
            if pts.size:
                centroids[i] = pts.mean(axis=0)

    sse = sum(np.linalg.norm(data_np[i] - centroids[labels[i]])**2 for i in range(len(data_np)))
    return labels, centroids, sse

sse_list, all_labels, all_centroids = [], [], []
for k in range(1, max_k + 1):
    labels, centroids, sse = compute_kmeans(data_np.copy(), k)
    sse_list.append(sse)
    all_labels.append(labels)
    all_centroids.append(centroids)

def elbow_method(ssecluster):
    x1, y1 = 0, ssecluster[0]
    x2, y2 = len(ssecluster)-1, ssecluster[-1]
    max_dist, opt_k = 0, 1
    for i in range(1, len(ssecluster)-1):
        x0, y0 = i, ssecluster[i]
        dist = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.hypot(y2-y1, x2-x1)
        if dist > max_dist:
            max_dist, opt_k = dist, i+1
    if opt_k > 2 and (ssecluster[opt_k-2] - ssecluster[opt_k-1]) / ssecluster[opt_k-2] < threshold:
        opt_k -= 1
    return opt_k

opt_k = elbow_method(sse_list)

st.markdown("---")
col1, col2 = st.columns([3, 5])
with col1:
    st.subheader("📈 Grafik SSE vs Kluster")
    fig = go.Figure(
        layout=go.Layout(
            template="plotly_white", title=dict(text="SSE vs Jumlah Kluster", x=0.5),
            xaxis=dict(title="Jumlah Kluster (K)", tickmode="linear", dtick=1, showgrid=False),
            yaxis=dict(title="SSE", showgrid=True, gridcolor="lightgrey"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
    )
    fig.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=sse_list, mode='lines+markers', marker=dict(size=8), line=dict(width=2)))
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🔎 Kluster optimal: {opt_k}")
with col2:
    st.subheader("📊 Hasil Clusterisasi Data")
    for i in range(opt_k):
        pts = data_np[all_labels[opt_k-1] == i]
        count = pts.shape[0]
        with st.expander(f"Cluster {i+1} ({count} data points)"):
            if count:
                df_cluster = pd.DataFrame(pts, columns=data_df.columns)
                st.dataframe(df_cluster, use_container_width=True)
            else:
                st.write("Tidak ada data pada kluster ini.")
# Footer
st.markdown("---")
st.markdown("© 2025 - Clustering KMeans | Alfin")