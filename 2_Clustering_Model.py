import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import get_feature_cols
from utils.clustering import (
    scale_features, reduce_dimensions, find_optimal_k,
    train_kmeans, train_agglomerative, evaluate_clusters,
    get_cluster_profiles, label_segments
)

st.set_page_config(page_title="Clustering", page_icon="🤖", layout="wide")

st.title("🤖 Clustering Model")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload data on the **Data Analysis** page first.")
    st.stop()

df = st.session_state["df"].copy()
feat_cols = get_feature_cols(df)

# ── Sidebar Controls ──────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
algorithm = st.sidebar.selectbox("Algorithm", ["K-Means", "Agglomerative Clustering"])
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)

if algorithm == "Agglomerative Clustering":
    linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
else:
    linkage = "ward"

selected_feats = st.sidebar.multiselect(
    "Features to include",
    feat_cols,
    default=[f for f in ["Income","TotalSpend","TotalPurchases","Age","Recency","Customer_Days"] if f in feat_cols]
)

if not selected_feats:
    st.warning("Select at least 2 features.")
    st.stop()

# ── Optimal K (Elbow + Silhouette) ───────────────────────────────
st.subheader("📐 Find Optimal Number of Clusters")
X_scaled, scaler = scale_features(df, selected_feats)

with st.expander("📊 Elbow Method & Silhouette Scores", expanded=True):
    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                                  line=dict(color="#667eea", width=3), marker=dict(size=8)))
        fig.update_layout(title="Elbow Method (Inertia vs K)",
                          xaxis_title="K (Clusters)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=silhouettes, mode="lines+markers",
                                  line=dict(color="#764ba2", width=3), marker=dict(size=8)))
        fig.add_vline(x=k_range[np.argmax(silhouettes)], line_dash="dash", line_color="red",
                      annotation_text=f"Best K={k_range[np.argmax(silhouettes)]}")
        fig.update_layout(title="Silhouette Score vs K",
                          xaxis_title="K (Clusters)", yaxis_title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)

# ── Train ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚀 Train Model")

if st.button("▶ Run Clustering", use_container_width=True):
    with st.spinner("Training..."):
        if algorithm == "K-Means":
            model, labels = train_kmeans(X_scaled, n_clusters)
        else:
            model, labels = train_agglomerative(X_scaled, n_clusters, linkage)

        metrics = evaluate_clusters(X_scaled, labels)
        df["Cluster"] = labels
        segment_names = label_segments(n_clusters)
        df["Segment"] = df["Cluster"].map(segment_names)

        # Store in session
        st.session_state["clustered_df"] = df
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["selected_feats"] = selected_feats
        st.session_state["segment_names"] = segment_names
        st.session_state["algorithm"] = algorithm
        st.session_state["n_clusters"] = n_clusters

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Silhouette Score", metrics["silhouette"], help="Higher is better (max 1.0)")
    m2.metric("📉 Davies-Bouldin", metrics["davies_bouldin"], help="Lower is better")
    m3.metric("📈 Calinski-Harabasz", metrics["calinski_harabasz"], help="Higher is better")

    st.success(f"✅ {algorithm} with K={n_clusters} trained successfully!")

    # Cluster profile
    st.markdown("---")
    st.subheader("📋 Cluster Profiles (Mean Values)")
    profile = get_cluster_profiles(df, selected_feats)
    profile.index = [f"Cluster {i} — {segment_names[i]}" for i in profile.index]
    st.dataframe(profile.style.background_gradient(cmap="Purples", axis=0), use_container_width=True)

    # Cluster size
    st.markdown("---")
    st.subheader("📊 Cluster Size Distribution")
    counts = df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment","Count"]
    fig = px.bar(counts, x="Segment", y="Count", color="Segment",
                 title="Customers per Cluster", text="Count")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif "clustered_df" in st.session_state:
    st.info("ℹ️ Model already trained. Go to **Visualizations** or **Predict Segment** page.")
