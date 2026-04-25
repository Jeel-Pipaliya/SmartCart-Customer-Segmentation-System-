# 🛒 SmartCart Customer Segmentation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcart-customer-segmentation.streamlit.app)

An intelligent customer segmentation system for SmartCart e-commerce platform using unsupervised machine learning (K-Means & Agglomerative Clustering).

---

## 📌 Features

| Page | Description |
|---|---|
| 📊 Data Analysis | Upload CSV, explore EDA, distributions, correlations |
| 🤖 Clustering Model | Train K-Means or Agglomerative, elbow method, silhouette scores |
| 📈 Visualizations | 2D/3D PCA plots, radar charts, heatmaps |
| 🎯 Predict Segment | Input new customer data → get predicted cluster + marketing tip |

## 🗂️ Project Structure

```
smartcart/
├── app.py                          # Home page
├── pages/
│   ├── 1_📊_Data_Analysis.py
│   ├── 2_🤖_Clustering_Model.py
│   ├── 3_📈_Visualizations.py
│   └── 4_🎯_Predict_Segment.py
├── utils/
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   └── clustering.py               # Model training & evaluation
├── .streamlit/
│   └── config.toml
└── requirements.txt
```

## 🚀 Run Locally

```bash
git clone https://github.com/Jeel-Pipaliya/SmartCart-Customer-Segmentation-System-.git
cd SmartCart-Customer-Segmentation-System-
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App**
4. Set:
   - **Repository**: `Jeel-Pipaliya/SmartCart-Customer-Segmentation-System-`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy** ✅

## 📊 Dataset

The app expects `smartcart_customers.csv` with 22 features:

**Demographics:** `ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer`

**Spending:** `MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds`

**Purchases:** `NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth`

**Other:** `Recency, Complain`

## 🔬 ML Pipeline

```
Raw CSV → Preprocessing → Feature Engineering → StandardScaler → PCA → K-Means / Agglomerative → Evaluation → Visualizations
```

**Engineered Features:** `Age, Customer_Days, TotalSpend, TotalPurchases, TotalChildren, Is_Partnered`

**Metrics Used:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

## 👤 Author

**Jeel Pipaliya** — AI/ML Engineer
