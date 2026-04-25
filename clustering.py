import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


def reduce_dimensions(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Return inertia and silhouette scores for elbow/silhouette plots."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, silhouettes


def train_kmeans(X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels


def train_agglomerative(X_scaled, n_clusters, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    return model, labels


def evaluate_clusters(X_scaled, labels):
    if len(np.unique(labels)) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": round(silhouette_score(X_scaled, labels), 4),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 2),
    }


def get_cluster_profiles(df, feature_cols, label_col="Cluster"):
    """Return per-cluster mean stats."""
    profile = df.groupby(label_col)[feature_cols].mean().round(2)
    profile["Count"] = df.groupby(label_col)[feature_cols[0]].count().values
    return profile


def label_segments(n_clusters):
    """Return simple human-readable segment names."""
    base = [
        "💎 Premium Loyalists",
        "🌱 Budget Shoppers",
        "🔥 High-Value Regulars",
        "😴 Dormant / At-Risk",
        "🛍️ Deal Hunters",
        "🧩 Casual Browsers",
    ]
    return {i: base[i % len(base)] for i in range(n_clusters)}
