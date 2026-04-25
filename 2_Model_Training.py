import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

st.set_page_config(page_title="Model Training", page_icon="🔬", layout="wide")
st.title("🔬 Model Training")
st.markdown("Encode → Scale → PCA → Find optimal K → Cluster")

# ─── Require preprocessed data ───────────────────────────────────────────────

if "df_clean" not in st.session_state:
    st.warning("⚠️ Please run **EDA & Preprocessing** first to load data.")
    st.stop()

df_clean = st.session_state["df_clean"].copy()

# ─── Encoding & Scaling ───────────────────────────────────────────────────────

@st.cache_data
def encode_and_scale(df):
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder(sparse_output=False)
    enc_arr = ohe.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df_enc = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_enc)
    return df_enc, X_scaled, ohe, scaler

df_encoded, X_scaled, ohe_model, scaler_model = encode_and_scale(df_clean)
st.session_state["df_encoded"] = df_encoded

st.success(f"✅ Encoded shape: {df_encoded.shape}  |  Scaled ✓")

# ─── PCA ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔻 PCA Dimensionality Reduction")

n_components = st.slider("Number of PCA components", 2, 10, 3)

@st.cache_data
def run_pca(X, n):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

X_pca, evr = run_pca(X_scaled, n_components)
st.session_state["X_pca"] = X_pca

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Variance Explained", f"{evr.sum()*100:.1f}%")
with col2:
    fig_evr, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, n_components + 1), evr * 100, color="#1f77b4")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance (%)")
    ax.set_title("Explained Variance per Component")
    st.pyplot(fig_evr)

# 3D PCA plot
if n_components >= 3:
    st.markdown("**3D PCA Scatter**")
    fig3d = plt.figure(figsize=(7, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.5, s=10, c="#1f77b4")
    ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
    ax3d.set_title("3D PCA Projection")
    st.pyplot(fig3d)

# ─── Optimal K ────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Find Optimal K")

k_max = st.slider("Max K to evaluate", 5, 15, 10)

@st.cache_data
def compute_k_metrics(X, k_max):
    wcss, sil = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        if k >= 2:
            sil.append(silhouette_score(X, labels))
    knee = KneeLocator(range(1, k_max + 1), wcss, curve="convex", direction="decreasing")
    return wcss, sil, knee.elbow

wcss, sil_scores, elbow_k = compute_k_metrics(X_pca, k_max)
best_sil_k = np.argmax(sil_scores) + 2  # offset since sil starts at k=2

col1, col2 = st.columns(2)
col1.metric("📐 Elbow K", elbow_k)
col2.metric("🏆 Best Silhouette K", best_sil_k)

fig_k, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, k_max + 1), wcss, marker="o", color="blue")
if elbow_k:
    ax1.axvline(elbow_k, color="red", linestyle="--", label=f"Elbow k={elbow_k}")
    ax1.legend()
ax1.set_xlabel("K"); ax1.set_ylabel("WCSS"); ax1.set_title("Elbow Method")

ax2.plot(range(2, k_max + 1), sil_scores, marker="o", color="green")
ax2.axvline(best_sil_k, color="red", linestyle="--", label=f"Best k={best_sil_k}")
ax2.legend()
ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Analysis")
plt.tight_layout()
st.pyplot(fig_k)

# ─── Clustering ───────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🎯 Train Clustering Models")

col1, col2 = st.columns(2)
with col1:
    n_clusters = st.number_input("Number of clusters (K)", min_value=2, max_value=10, value=4)
with col2:
    algorithm = st.selectbox("Algorithm", ["KMeans", "Agglomerative (Ward)", "Both"])

if st.button("🚀 Run Clustering"):
    pal = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"]

    results = {}

    if algorithm in ["KMeans", "Both"]:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_km = km.fit_predict(X_pca)
        sil_km = silhouette_score(X_pca, labels_km)
        results["KMeans"] = (labels_km, sil_km)

    if algorithm in ["Agglomerative (Ward)", "Both"]:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels_agg = agg.fit_predict(X_pca)
        sil_agg = silhouette_score(X_pca, labels_agg)
        results["Agglomerative"] = (labels_agg, sil_agg)

    # Store best labels
    best_algo = max(results, key=lambda k: results[k][1])
    best_labels = results[best_algo][0]
    st.session_state["cluster_labels"] = best_labels
    st.session_state["best_algo"] = best_algo
    st.session_state["n_clusters"] = n_clusters

    # Metrics
    st.markdown("### 📊 Results")
    metric_cols = st.columns(len(results))
    for idx, (algo, (labels, sil)) in enumerate(results.items()):
        metric_cols[idx].metric(f"{algo} Silhouette", f"{sil:.4f}",
                                delta="✅ Best" if algo == best_algo else None)

    # 3D plots
    n_plots = len(results)
    fig_cluster = plt.figure(figsize=(8 * n_plots, 6))
    for idx, (algo, (labels, sil)) in enumerate(results.items()):
        ax = fig_cluster.add_subplot(1, n_plots, idx + 1, projection="3d")
        for c in range(n_clusters):
            mask = labels == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2] if X_pca.shape[1] > 2 else np.zeros(mask.sum()),
                       label=f"Cluster {c}", s=15, alpha=0.7, color=pal[c % len(pal)])
        ax.set_title(f"{algo}\nSilhouette={sil:.3f}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_cluster)

    st.success(f"✅ Best model: **{best_algo}** (Silhouette = {results[best_algo][1]:.4f})")
    st.info("Navigate to **Cluster Analysis** to explore segments.")

elif "cluster_labels" in st.session_state:
    st.info(f"✅ Previously trained with **{st.session_state.get('best_algo')}** — {st.session_state.get('n_clusters')} clusters. Navigate to Cluster Analysis.")
