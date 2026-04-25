import streamlit as st

st.set_page_config(
    page_title="SmartCart Clustering System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛒 SmartCart Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Customer Clustering using Unsupervised Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**📊 EDA & Preprocessing**\n\nExplore raw data, handle missing values, feature engineering, and outlier removal.")
with col2:
    st.info("**🔬 Model Training**\n\nPCA dimensionality reduction, Elbow + Silhouette analysis, KMeans & Agglomerative Clustering.")
with col3:
    st.info("**📈 Cluster Analysis**\n\nVisualise clusters, interpret segments, and download results.")

st.markdown("---")
st.markdown("### 🚀 Getting Started")
st.markdown("""
Use the **sidebar** to navigate between pages:

1. **📊 EDA & Preprocessing** — Upload or use default dataset, explore distributions, handle missing values, engineer features.
2. **🔬 Model Training** — Run PCA, determine optimal K, train clustering models.
3. **📈 Cluster Analysis** — Deep-dive into each cluster's characteristics.

> All results are cached — navigate freely between pages without re-running computations.
""")

st.markdown("---")
with st.expander("📋 Dataset Overview"):
    st.markdown("""
    | Category | Features |
    |---|---|
    | **Demographics** | Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer |
    | **Amount Spent** | MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds |
    | **Purchase Frequency** | NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth |
    | **Feedback** | Recency, Complain, Response |
    """)
