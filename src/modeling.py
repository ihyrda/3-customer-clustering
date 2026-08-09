"""K-means evaluation, selection, profiling, and customer assignment.

fit_kmeans - fit one reproducible candidate
evaluate_cluster_counts - compare k=3 through k=8
select_cluster_count - select the highest silhouette score
create_cluster_profiles - summarize fitted groups in original units
assign_customer - place one customer using the fitted scaler and model
train_clustering_workflow - run both notebook and final feature views
"""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import final_feats
from src.preprocessing import (
    get_three_feature_data,
    get_two_feature_data,
    scale_features,
)

logger = logging.getLogger(__name__)


def fit_kmeans(X, n_clusters: int) -> KMeans:
    """Fit one K-means model with reproducible initialization."""
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42,
    )
    return model.fit(X)


def evaluate_cluster_counts(X, k_values=range(3, 9)) -> pd.DataFrame:
    """Return inertia and silhouette score for every candidate k."""
    rows = []
    for k in k_values:
        model = fit_kmeans(X, k)
        rows.append({
            "Clusters": int(k),
            "Inertia": float(model.inertia_),
            "Silhouette": float(silhouette_score(X, model.labels_)),
        })
        logger.info(
            "k=%d: inertia %.3f, silhouette %.4f",
            k,
            rows[-1]["Inertia"],
            rows[-1]["Silhouette"],
        )
    return pd.DataFrame(rows)


def select_cluster_count(metrics: pd.DataFrame) -> int:
    """Select k with the highest silhouette score."""
    selected = int(metrics.loc[metrics["Silhouette"].idxmax(), "Clusters"])
    logger.info("Selected k=%d", selected)
    return selected


def create_cluster_profiles(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize fitted clusters in original feature units."""
    profiles = (
        assignments.groupby("Cluster")[final_feats]
        .mean()
        .round(2)
        .reset_index()
    )
    sizes = assignments.groupby("Cluster").size().reset_index(name="Customers")
    profiles = profiles.merge(sizes, on="Cluster")
    profiles["Share"] = profiles["Customers"] / len(assignments)
    return profiles


def assign_customer(
    model: KMeans,
    scaler: StandardScaler,
    customer_df: pd.DataFrame,
) -> int:
    """Assign one customer in the fitted model's scaled feature space."""
    scaled = scaler.transform(customer_df)
    cluster = int(model.predict(scaled)[0])
    logger.info("Assigned new customer to cluster %d", cluster)
    return cluster


def train_clustering_workflow(df: pd.DataFrame) -> dict:
    """Run the raw notebook view and the scaled final clustering model."""
    two_feature_data = get_two_feature_data(df)
    two_feature_metrics = evaluate_cluster_counts(two_feature_data)
    two_feature_k = select_cluster_count(two_feature_metrics)

    final_data = get_three_feature_data(df)
    scaler, scaled_data = scale_features(final_data)
    final_metrics = evaluate_cluster_counts(scaled_data)
    selected_k = select_cluster_count(final_metrics)
    model = fit_kmeans(scaled_data, selected_k)

    assignments = df.copy()
    assignments["Cluster"] = model.labels_
    profiles = create_cluster_profiles(assignments)

    logger.info("Clustering complete: two-feature k=%d, final k=%d", two_feature_k, selected_k)
    return {
        "two_feature_metrics": two_feature_metrics,
        "two_feature_k": two_feature_k,
        "final_metrics": final_metrics,
        "selected_k": selected_k,
        "model": model,
        "scaler": scaler,
        "assignments": assignments,
        "profiles": profiles,
    }
