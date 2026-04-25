import pandas as pd
import numpy as np
from datetime import datetime


def load_and_preprocess(file):
    """Load CSV and apply full preprocessing pipeline."""
    df = pd.read_csv(file)
    df = df.copy()

    # --- 1. Drop duplicates & irrelevant columns ---
    df.drop_duplicates(inplace=True)
    drop_cols = [c for c in ["ID", "Z_CostContact", "Z_Revenue", "Response",
                              "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                              "AcceptedCmp4","AcceptedCmp5"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # --- 2. Handle missing Income ---
    if "Income" in df.columns:
        df["Income"].fillna(df["Income"].median(), inplace=True)

    # --- 3. Feature Engineering ---
    current_year = datetime.now().year
    if "Year_Birth" in df.columns:
        df["Age"] = current_year - df["Year_Birth"]
        df.drop(columns=["Year_Birth"], inplace=True)

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
        df["Customer_Days"] = (datetime.now() - df["Dt_Customer"]).dt.days
        df.drop(columns=["Dt_Customer"], inplace=True)

    # Total spending
    spend_cols = [c for c in ["MntWines","MntFruits","MntMeatProducts",
                               "MntFishProducts","MntSweetProducts","MntGoldProds"]
                  if c in df.columns]
    if spend_cols:
        df["TotalSpend"] = df[spend_cols].sum(axis=1)

    # Total purchases
    purchase_cols = [c for c in ["NumDealsPurchases","NumWebPurchases",
                                  "NumCatalogPurchases","NumStorePurchases"]
                     if c in df.columns]
    if purchase_cols:
        df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    # Children
    kid_cols = [c for c in ["Kidhome","Teenhome"] if c in df.columns]
    if kid_cols:
        df["TotalChildren"] = df[kid_cols].sum(axis=1)

    # --- 4. Encode categoricals ---
    if "Education" in df.columns:
        edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
        df["Education"] = df["Education"].map(edu_map).fillna(2)

    if "Marital_Status" in df.columns:
        df["Is_Partnered"] = df["Marital_Status"].apply(
            lambda x: 1 if str(x).strip() in ["Married","Together"] else 0
        )
        df.drop(columns=["Marital_Status"], inplace=True)

    # --- 5. Remove outliers (IQR on Age & Income) ---
    for col in ["Age", "Income"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            df = df[(df[col] >= Q1) & (df[col] <= Q3)]

    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_cols(df):
    """Return numeric columns suitable for clustering."""
    exclude = ["Cluster"]
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
