import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.clustering import scale_features

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

st.title("📈 Cluster Visualizations")

if "clustered_df" not in st.session_state:
    st.warning("⚠️ Please train a model on the **Clustering Model** page first.")
    st.stop()

df = st.session_state["clustered_df"].copy()
selected_feats = st.session_state["selected_feats"]
segment_names = st.session_state["segment_names"]
n_clusters = st.session_state["n_clusters"]

X_scaled, _ = scale_features(df, selected_feats)

# ── PCA 2D Scatter ─────────────────────────────────────────────────
st.subheader("🔵 2D Cluster Scatter (PCA)")
pca2 = PCA(n_components=2, random_state=42)
coords2 = pca2.fit_transform(X_scaled)
pca_df = pd.DataFrame(coords2, columns=["PC1","PC2"])
pca_df["Cluster"] = df["Cluster"].values.astype(str)
pca_df["Segment"] = df["Segment"].values
pca_df["Income"] = df["Income"].values if "Income" in df.columns else 0
pca_df["TotalSpend"] = df["TotalSpend"].values if "TotalSpend" in df.columns else 0

fig = px.scatter(pca_df, x="PC1", y="PC2", color="Segment",
                 hover_data=["Income","TotalSpend"],
                 title=f"Customer Clusters in 2D PCA Space (Explained Variance: {sum(pca2.explained_variance_ratio_)*100:.1f}%)",
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_traces(marker=dict(size=5, opacity=0.7))
st.plotly_chart(fig, use_container_width=True)

# ── PCA 3D Scatter ─────────────────────────────────────────────────
if len(selected_feats) >= 3:
    st.subheader("🌐 3D Cluster Scatter (PCA)")
    pca3 = PCA(n_components=3, random_state=42)
    coords3 = pca3.fit_transform(X_scaled)
    pca3_df = pd.DataFrame(coords3, columns=["PC1","PC2","PC3"])
    pca3_df["Segment"] = df["Segment"].values
    fig3 = px.scatter_3d(pca3_df, x="PC1", y="PC2", z="PC3", color="Segment",
                         title="3D PCA Cluster View", opacity=0.7,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)

# ── Radar Chart ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🕸️ Cluster Profile Radar Chart")

profile = df.groupby("Cluster")[selected_feats].mean()

# Normalize 0-1 for radar
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
profile_norm = pd.DataFrame(norm.fit_transform(profile), columns=selected_feats, index=profile.index)

fig_radar = go.Figure()
colors = px.colors.qualitative.Set2
for i, row in profile_norm.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=row.values.tolist() + [row.values[0]],
        theta=selected_feats + [selected_feats[0]],
        fill="toself",
        name=segment_names.get(i, f"Cluster {i}"),
        line=dict(color=colors[i % len(colors)])
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                         showlegend=True, title="Normalized Cluster Profiles")
st.plotly_chart(fig_radar, use_container_width=True)

# ── Feature Boxplots ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📦 Feature Distribution per Cluster")

feat_select = st.selectbox("Select feature:", selected_feats)
fig_box = px.box(df, x="Segment", y=feat_select, color="Segment",
                 title=f"{feat_select} by Customer Segment",
                 color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig_box, use_container_width=True)

# ── Heatmap ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔥 Cluster Feature Heatmap")
profile_display = profile.T
profile_display.columns = [segment_names.get(c, f"C{c}") for c in profile_display.columns]
fig_heat = px.imshow(profile_display, text_auto=".0f", aspect="auto",
                     color_continuous_scale="Viridis",
                     title="Average Feature Values per Cluster")
st.plotly_chart(fig_heat, use_container_width=True)

# ── Download ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⬇️ Download Clustered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV with Cluster Labels", data=csv,
                   file_name="smartcart_clustered.csv", mime="text/csv",
                   use_container_width=True)
