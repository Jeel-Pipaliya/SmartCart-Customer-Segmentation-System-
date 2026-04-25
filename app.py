import streamlit as st

st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛒 SmartCart Customer Segmentation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent customer clustering using unsupervised machine learning</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("📊 **Data Analysis**\nExplore dataset statistics and distributions")
with col2:
    st.info("🔍 **Clustering**\nApply K-Means & Agglomerative algorithms")
with col3:
    st.info("📈 **Visualization**\nInteractive cluster visualizations")
with col4:
    st.info("🎯 **Prediction**\nPredict segment for new customers")

st.markdown("---")

st.markdown("### 📌 How to Use")
st.markdown("""
1. **Upload Data** → Go to *Data Analysis* page and upload `smartcart_customers.csv`
2. **Explore EDA** → Understand distributions, missing values, and correlations
3. **Train Clusters** → Go to *Clustering* page, select algorithm & number of clusters
4. **Visualize Results** → See cluster plots and segment profiles
5. **Predict** → Enter new customer data and get their segment on *Prediction* page
""")

st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("""
Use the **pages** above to navigate:
- 📊 Data Analysis
- 🤖 Clustering Model
- 📈 Visualizations
- 🎯 Predict Segment
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**SmartCart v1.0**\nBuilt with Streamlit + Scikit-learn")
