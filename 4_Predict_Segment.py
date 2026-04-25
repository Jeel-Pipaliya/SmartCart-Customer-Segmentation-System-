import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import get_feature_cols

st.set_page_config(page_title="Predict Segment", page_icon="🎯", layout="wide")

st.title("🎯 Predict Customer Segment")
st.markdown("Enter a new customer's details to predict their segment.")

if "clustered_df" not in st.session_state or "scaler" not in st.session_state:
    st.warning("⚠️ Please train a model on the **Clustering Model** page first.")
    st.stop()

df = st.session_state["clustered_df"]
scaler = st.session_state["scaler"]
model = st.session_state["model"]
selected_feats = st.session_state["selected_feats"]
segment_names = st.session_state["segment_names"]
algorithm = st.session_state["algorithm"]

st.info(f"🧠 Active model: **{algorithm}** | Features used: `{', '.join(selected_feats)}`")

st.markdown("---")
st.subheader("📝 Enter Customer Details")

# Dynamically build input fields based on selected features
input_data = {}
col_pairs = [selected_feats[i:i+3] for i in range(0, len(selected_feats), 3)]

for trio in col_pairs:
    cols = st.columns(len(trio))
    for col, feat in zip(cols, trio):
        with col:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            input_data[feat] = st.number_input(
                feat,
                min_value=min_val,
                max_value=max_val,
                value=round(mean_val, 2),
                step=(max_val - min_val) / 100,
                format="%.2f"
            )

st.markdown("---")

if st.button("🔍 Predict My Segment", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[selected_feats])

    if algorithm == "K-Means":
        cluster_id = int(model.predict(input_scaled)[0])
    else:
        # Agglomerative doesn't have predict — use nearest centroid from training data
        from sklearn.metrics.pairwise import euclidean_distances
        X_train_scaled = scaler.transform(df[selected_feats])
        centroids = np.array([X_train_scaled[df["Cluster"] == c].mean(axis=0)
                               for c in sorted(df["Cluster"].unique())])
        cluster_id = int(np.argmin(euclidean_distances(input_scaled, centroids)))

    segment = segment_names.get(cluster_id, f"Cluster {cluster_id}")

    st.success(f"### ✅ Predicted Segment: {segment}")

    # Show cluster stats comparison
    st.markdown("---")
    st.subheader("📊 Your Cluster vs Other Clusters")

    compare = df.groupby("Cluster")[selected_feats].mean().round(2)
    compare.index = [segment_names.get(i, f"Cluster {i}") for i in compare.index]

    # Highlight the predicted cluster
    def highlight_row(row):
        return ["background-color: #d4edda; font-weight: bold;" if row.name == segment else "" for _ in row]

    st.dataframe(compare.style.apply(highlight_row, axis=1), use_container_width=True)

    # Radar for this customer vs cluster average
    st.markdown("---")
    st.subheader("🕸️ Your Profile vs Cluster Average")

    import plotly.graph_objects as go
    from sklearn.preprocessing import MinMaxScaler

    cluster_avg = df[df["Cluster"] == cluster_id][selected_feats].mean()
    user_vals = pd.Series(input_data)[selected_feats]

    mm = MinMaxScaler()
    combined = pd.DataFrame([cluster_avg.values, user_vals.values], columns=selected_feats)
    combined_norm = pd.DataFrame(mm.fit_transform(combined), columns=selected_feats)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=combined_norm.iloc[0].tolist() + [combined_norm.iloc[0].iloc[0]],
        theta=selected_feats + [selected_feats[0]],
        fill="toself", name=f"Cluster Avg ({segment})",
        line=dict(color="#667eea")
    ))
    fig.add_trace(go.Scatterpolar(
        r=combined_norm.iloc[1].tolist() + [combined_norm.iloc[1].iloc[0]],
        theta=selected_feats + [selected_feats[0]],
        fill="toself", name="Your Profile",
        line=dict(color="#f64f59", dash="dash")
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      title="Customer Profile vs Cluster Average")
    st.plotly_chart(fig, use_container_width=True)

    # Marketing recommendation
    st.markdown("---")
    st.subheader("💡 Marketing Recommendation")

    recommendations = {
        0: "🎁 Offer exclusive loyalty rewards, early access to new products, and personalized premium offers.",
        1: "💵 Target with budget deals, combo packs, and discount campaigns to increase basket size.",
        2: "📧 Send personalized email campaigns with cross-sell suggestions for complementary products.",
        3: "🔔 Re-engagement campaigns with win-back offers, 'We miss you' discounts, and reminders.",
        4: "🏷️ Promote flash sales, coupons, and deal bundles to satisfy discount-seeking behaviour.",
        5: "📲 Nurture with content marketing, product discovery emails, and soft conversion nudges.",
    }
    msg = recommendations.get(cluster_id % len(recommendations), "📌 Targeted engagement recommended.")
    st.info(msg)
