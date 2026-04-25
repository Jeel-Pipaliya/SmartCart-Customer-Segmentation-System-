import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="EDA & Preprocessing", page_icon="📊", layout="wide")

st.title("📊 EDA & Preprocessing")
st.markdown("Explore, clean, and engineer features from the SmartCart dataset.")

# ─── Load Data ───────────────────────────────────────────────────────────────

def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("smartcart_customers.csv")
    return df

with st.sidebar:
    st.header("📁 Data Source")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    use_default = st.checkbox("Use default dataset", value=True)

if uploaded:
    raw_df = load_data(uploaded)
    st.success("✅ Custom dataset loaded!")
elif use_default:
    raw_df = load_data()
    st.info("ℹ️ Using default smartcart_customers.csv")
else:
    st.warning("Please upload a CSV or enable the default dataset.")
    st.stop()

# ─── Raw Data ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🗂️ Raw Data Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", raw_df.shape[0])
col2.metric("Columns", raw_df.shape[1])
col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

st.dataframe(raw_df.head(10), use_container_width=True)

# ─── Missing Values ───────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Missing Values")
missing = raw_df.isnull().sum()
missing = missing[missing > 0].reset_index()
missing.columns = ["Feature", "Missing Count"]
if missing.empty:
    st.success("No missing values found — except Income (if any).")
else:
    st.dataframe(missing, use_container_width=True)

# ─── Preprocessing ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("⚙️ Feature Engineering")

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Fill missing Income
    df["Income"] = df["Income"].fillna(df["Income"].median())
    # Age
    df["Age"] = 2026 - df["Year_Birth"]
    # Tenure
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (ref - df["Dt_Customer"]).dt.days
    # Spending
    df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
                            + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
    # Children
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    # Education
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    # Marital
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    # Drop columns
    drop_cols = (["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
                 + ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"])
    df_cleaned = df.drop(columns=drop_cols)
    # Remove outliers
    df_cleaned = df_cleaned[(df_cleaned["Age"] < 90) & (df_cleaned["Income"] < 600_000)]
    return df_cleaned

df_clean = preprocess(raw_df)
st.session_state["df_clean"] = df_clean

with st.expander("View engineered features"):
    st.dataframe(df_clean.head(10), use_container_width=True)
    st.write(f"Shape after preprocessing: {df_clean.shape}")

# ─── Distributions ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📈 Feature Distributions")

num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
selected = st.multiselect("Select features to plot", num_cols,
                           default=["Income", "Total_Spending", "Age", "Recency"])

if selected:
    n = len(selected)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, min(n, cols_per_row), figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(selected):
        sns.histplot(df_clean[col], ax=axes[i], kde=True, color="#1f77b4")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# ─── Categorical Distributions ────────────────────────────────────────────────

st.markdown("---")
st.subheader("🏷️ Categorical Features")
cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
if cat_cols:
    fig2, axes2 = plt.subplots(1, len(cat_cols), figsize=(12, 4))
    if len(cat_cols) == 1:
        axes2 = [axes2]
    for ax, col in zip(axes2, cat_cols):
        vc = df_clean[col].value_counts()
        ax.bar(vc.index, vc.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig2)

# ─── Correlation Heatmap ──────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🌡️ Correlation Heatmap")
corr = df_clean.corr(numeric_only=True)
fig3, ax3 = plt.subplots(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, annot_kws={"size": 6}, cmap="coolwarm",
            mask=mask, ax=ax3, fmt=".2f", linewidths=0.5)
ax3.set_title("Feature Correlation Matrix")
plt.tight_layout()
st.pyplot(fig3)

# ─── Download ─────────────────────────────────────────────────────────────────

st.markdown("---")
csv_buf = io.StringIO()
df_clean.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download Preprocessed Data", csv_buf.getvalue(),
                   "smartcart_preprocessed.csv", "text/csv")

st.success("✅ Preprocessing complete! Navigate to **Model Training** to cluster customers.")
