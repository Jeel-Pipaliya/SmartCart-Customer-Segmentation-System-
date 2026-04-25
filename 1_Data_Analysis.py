import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_preprocess, get_feature_cols

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

st.title("📊 Data Analysis & EDA")
st.markdown("Upload your SmartCart customer CSV to explore the data.")

uploaded = st.file_uploader("Upload `smartcart_customers.csv`", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)
    df = load_and_preprocess(uploaded)
    st.session_state["df"] = df
    st.session_state["raw_df"] = raw_df

    st.success(f"✅ Loaded {len(raw_df)} records → After preprocessing: **{len(df)} clean records**")

    # ── Overview ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(raw_df))
    c2.metric("Features", raw_df.shape[1])
    c3.metric("Missing Values", int(raw_df.isnull().sum().sum()))
    c4.metric("After Cleaning", len(df))

    tab1, tab2, tab3, tab4 = st.tabs(["Raw Preview", "Processed Preview", "Statistics", "Missing Values"])

    with tab1:
        st.dataframe(raw_df.head(20), use_container_width=True)

    with tab2:
        st.dataframe(df.head(20), use_container_width=True)

    with tab3:
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    with tab4:
        miss = raw_df.isnull().sum()
        miss = miss[miss > 0].reset_index()
        miss.columns = ["Column", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(raw_df) * 100).round(2)
        if miss.empty:
            st.success("No missing values found!")
        else:
            fig = px.bar(miss, x="Column", y="Missing %", title="Missing Value % per Column",
                         color="Missing %", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Feature Distributions")
    feat_cols = get_feature_cols(df)
    selected = st.selectbox("Select feature to visualize:", feat_cols, index=feat_cols.index("TotalSpend") if "TotalSpend" in feat_cols else 0)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=selected, nbins=40, title=f"Distribution of {selected}",
                           color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, y=selected, title=f"Box Plot: {selected}",
                     color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔗 Correlation Heatmap")
    top_feats = feat_cols[:15]
    corr = df[top_feats].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    # ── Spend breakdown ───────────────────────────────────────────
    spend_cols = [c for c in ["MntWines","MntFruits","MntMeatProducts",
                               "MntFishProducts","MntSweetProducts","MntGoldProds"] if c in df.columns]
    if spend_cols:
        st.markdown("---")
        st.subheader("💰 Spending Category Breakdown")
        spend_means = df[spend_cols].mean().reset_index()
        spend_means.columns = ["Category", "Avg Spend"]
        fig = px.pie(spend_means, values="Avg Spend", names="Category",
                     title="Average Spending by Category", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload the CSV file to get started.")
    st.markdown("""
    **Expected columns include:**
    `ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer,
    MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
    NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases,
    NumWebVisitsMonth, Recency, Complain`
    """)
