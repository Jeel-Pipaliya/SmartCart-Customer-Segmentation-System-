# 🛒 SmartCart Customer Segmentation System

An intelligent customer clustering system built with **Streamlit** and **Scikit-learn**.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcart-customer-segmentation.streamlit.app)

## 📋 Features

| Page | Description |
|---|---|
| 🏠 Home | Project overview and navigation guide |
| 📊 EDA & Preprocessing | Data exploration, feature engineering, outlier removal |
| 🔬 Model Training | PCA, Elbow Method, Silhouette Analysis, KMeans & Agglomerative |
| 📈 Cluster Analysis | Segment visualisations, radar charts, business recommendations |

## 🗂️ Dataset
- 2240 customer records, 22 attributes
- Demographics, purchase behaviour, website activity

## ⚙️ Tech Stack
- **Streamlit** — UI
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — PCA, KMeans, AgglomerativeClustering
- **Seaborn / Matplotlib** — Visualisations
- **Kneed** — Automatic elbow detection

## 🛠️ Run Locally

```bash
git clone https://github.com/Jeel-Pipaliya/SmartCart-Customer-Segmentation-System-.git
cd SmartCart-Customer-Segmentation-System-
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select repo → set **Main file path** to `app.py`
4. Click **Deploy**

## 📁 Project Structure

```
├── app.py                          # Home page
├── pages/
│   ├── 1_📊_EDA_&_Preprocessing.py
│   ├── 2_🔬_Model_Training.py
│   └── 3_📈_Cluster_Analysis.py
├── smartcart_customers.csv         # Dataset
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Dark theme config
└── README.md
```
