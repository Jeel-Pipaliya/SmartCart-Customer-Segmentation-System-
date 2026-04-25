import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Cluster Analysis", page_icon="📈", layout="wide")
st.title("📈 Cluster Analysis & Insights")
st.markdown("Deep-dive into each customer segment to derive actionable business insights.")

# ─── Require data ─────────────────────────────────────────────────────────────

if "df_encoded" not in st.session_state or "cluster_labels" not in st.session_state:
    st.warning("⚠️ Please complete **EDA & Preprocessing** and **Model Training** first.")
    st.stop()

df = st.session_state["df_encoded"].copy()
labels = st.session_state["cluster_labels"]
algo = st.session_state.get("best_algo", "Model")
n_clusters = st.session_state.get("n_clusters", len(np.unique(labels)))

df["Cluster"] = labels

PALETTE = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71",
           "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

CLUSTER_NAMES = {
    0: "💎 High-Value Loyalists",
    1: "🌱 Budget-Conscious Buyers",
    2: "⚡ Engaged Mid-Spenders",
    3: "😴 Low-Engagement / At-Risk",
    4: "🔥 Premium Deal Seekers",
    5: "🧩 Niche Enthusiasts",
}

# ─── Overview ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader(f"📊 Cluster Distribution — {algo}")

col_metrics = st.columns(n_clusters)
for i in range(n_clusters):
    cnt = (df["Cluster"] == i).sum()
    pct = cnt / len(df) * 100
    name = CLUSTER_NAMES.get(i, f"Cluster {i}")
    col_metrics[i].metric(name, f"{cnt}", f"{pct:.1f}%")

fig_dist, ax = plt.subplots(figsize=(8, 4))
counts = df["Cluster"].value_counts().sort_index()
bars = ax.bar([CLUSTER_NAMES.get(i, f"Cluster {i}") for i in counts.index],
              counts.values, color=[PALETTE[i % len(PALETTE)] for i in counts.index])
ax.bar_label(bars, fmt="%d")
ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
ax.set_title("Customer Count per Cluster")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
st.pyplot(fig_dist)

# ─── Cluster Summary Table ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Cluster Summary Statistics")

numeric_cols = df.select_dtypes(include=np.number).drop(columns=["Cluster"]).columns.tolist()
summary = df.groupby("Cluster")[numeric_cols].mean().round(2)
summary.index = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in summary.index]

st.dataframe(summary.T, use_container_width=True)

# ─── Income vs Spending ───────────────────────────────────────────────────────

st.markdown("---")
st.subheader("💰 Income vs Total Spending")

if "Income" in df.columns and "Total_Spending" in df.columns:
    fig_is, ax_is = plt.subplots(figsize=(9, 5))
    for i in range(n_clusters):
        mask = df["Cluster"] == i
        ax_is.scatter(df.loc[mask, "Total_Spending"], df.loc[mask, "Income"],
                      label=CLUSTER_NAMES.get(i, f"Cluster {i}"),
                      color=PALETTE[i % len(PALETTE)], alpha=0.6, s=20)
    ax_is.set_xlabel("Total Spending"); ax_is.set_ylabel("Income")
    ax_is.set_title("Income vs Total Spending by Cluster")
    ax_is.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_is)

# ─── Feature Comparison ───────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔎 Feature Deep-Dive")

key_features = [c for c in ["Income", "Total_Spending", "Age", "Recency",
                              "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases",
                              "NumDealsPurchases", "Total_Children", "Customer_Tenure_Days"]
                if c in df.columns]

selected_feat = st.multiselect("Select features to compare across clusters", key_features,
                                default=key_features[:4])

if selected_feat:
    n = len(selected_feat)
    fig_feat, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, selected_feat):
        data = [df.loc[df["Cluster"] == i, feat].dropna().values for i in range(n_clusters)]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
        ax.set_xticklabels([f"C{i}" for i in range(n_clusters)])
        ax.set_title(feat); ax.set_xlabel("Cluster")
    plt.tight_layout()
    st.pyplot(fig_feat)

# ─── Radar Chart ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🕸️ Cluster Radar Chart")

radar_features = [c for c in ["Income", "Total_Spending", "Recency", "Age",
                                "NumWebPurchases", "NumStorePurchases", "Total_Children"]
                  if c in df.columns]

if len(radar_features) >= 3:
    cluster_means = df.groupby("Cluster")[radar_features].mean()
    # Normalise 0-1
    norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_r = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i in range(n_clusters):
        vals = norm.iloc[i].tolist()
        vals += vals[:1]
        ax_r.plot(angles, vals, color=PALETTE[i % len(PALETTE)], linewidth=2,
                  label=CLUSTER_NAMES.get(i, f"Cluster {i}"))
        ax_r.fill(angles, vals, color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_features, fontsize=9)
    ax_r.set_title("Cluster Radar Chart (normalised)", size=13, pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_radar)

# ─── Business Recommendations ────────────────────────────────────────────────

st.markdown("---")
st.subheader("💡 Business Recommendations")

recommendations = {
    0: ("💎 High-Value Loyalists",
        "High income, high spending. These are your best customers.",
        ["Offer VIP loyalty programmes and exclusive early access.",
         "Personalise premium product recommendations.",
         "Assign dedicated account managers or concierge service."]),
    1: ("🌱 Budget-Conscious Buyers",
        "Lower income/spending, deal-sensitive.",
        ["Target with discount campaigns and bundle deals.",
         "Promote value-for-money product lines.",
         "Use email drip campaigns with coupon codes."]),
    2: ("⚡ Engaged Mid-Spenders",
        "Moderate spending, highly active on web.",
        ["Push cross-sell and upsell recommendations.",
         "Gamify engagement with points and rewards.",
         "Re-target with personalised web push notifications."]),
    3: ("😴 Low-Engagement / At-Risk",
        "Low recency and spending — potential churn.",
        ["Send win-back emails with strong incentives.",
         "Conduct survey to understand drop-off reasons.",
         "Offer time-limited reactivation discounts."]),
}

for cluster_idx in range(n_clusters):
    if cluster_idx in recommendations:
        name, desc, actions = recommendations[cluster_idx]
        with st.expander(f"{name} — Cluster {cluster_idx}"):
            st.markdown(f"**Profile:** {desc}")
            st.markdown("**Recommended Actions:**")
            for action in actions:
                st.markdown(f"- {action}")
    else:
        with st.expander(f"Cluster {cluster_idx}"):
            st.markdown("Analyse the summary table above to define a strategy for this segment.")

# ─── Download ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("⬇️ Export Results")

df_export = df.copy()
df_export["Cluster_Name"] = df_export["Cluster"].map(
    lambda i: CLUSTER_NAMES.get(i, f"Cluster {i}"))

buf = io.StringIO()
df_export.to_csv(buf, index=False)
st.download_button("⬇️ Download Clustered Data (CSV)", buf.getvalue(),
                   "smartcart_clustered.csv", "text/csv")

summary_buf = io.StringIO()
summary.to_csv(summary_buf)
st.download_button("⬇️ Download Cluster Summary (CSV)", summary_buf.getvalue(),
                   "smartcart_cluster_summary.csv", "text/csv")
