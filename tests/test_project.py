"""Focused tests for the customer-clustering project.

Run from the project root, inside the activated virtual environment:

    python -m pytest -v

The dataset and complete clustering result bundle are session fixtures so the
candidate models are not repeatedly fitted.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import final_feats, required_cols, load_data, validate_data
from src.modeling import (
    assign_customer,
    fit_kmeans,
    select_cluster_count,
    train_clustering_workflow,
)
from src.preprocessing import (
    get_three_feature_data,
    get_two_feature_data,
    prepare_customer_input,
)


@pytest.fixture(scope="session")
def df():
    """The validated course dataset."""
    frame = load_data()
    validate_data(frame)
    return frame


@pytest.fixture(scope="session")
def results(df):
    """The complete fitted clustering result bundle."""
    return train_clustering_workflow(df)


@pytest.fixture
def customer_values():
    """One complete set of app input values."""
    return {"Age": 35, "Annual_Income": 60, "Spending_Score": 50}


# --- data ------------------------------------------------------------------

# The bundled CSV is the expected course dataset.
def test_dataset_loads_with_expected_shape_and_columns(df):
    assert df.shape == (200, 5)
    assert list(df.columns) == required_cols


# A missing file raises instead of returning an empty frame.
def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


# A swapped or incomplete source file is rejected.
def test_wrong_columns_raise_error(df):
    broken = df.rename(columns={"Age": "Customer_Age"})
    with pytest.raises(ValueError, match="Expected columns"):
        validate_data(broken)


# Numeric source fields remain numeric clustering coordinates.
def test_non_numeric_feature_raises_error(df):
    broken = df.copy()
    broken["Age"] = broken["Age"].astype(str)
    with pytest.raises(ValueError, match="non-numeric"):
        validate_data(broken)


# --- preprocessing ---------------------------------------------------------

# The two notebook views exclude the identifier and descriptive category.
def test_feature_views_match_notebook_order(df):
    assert list(get_two_feature_data(df).columns) == [
        "Annual_Income",
        "Spending_Score",
    ]
    assert list(get_three_feature_data(df).columns) == final_feats


# The final scaler is fitted on all three source features in frozen order.
def test_final_scaler_matches_source_features(df, results):
    expected_means = get_three_feature_data(df).mean().to_numpy()
    assert np.allclose(results["scaler"].mean_, expected_means)
    assert results["model"].n_features_in_ == len(final_feats)


# The app dictionary becomes exactly the three final-model columns.
def test_customer_input_has_expected_columns(customer_values):
    row = prepare_customer_input(customer_values)
    assert row.shape == (1, 3)
    assert list(row.columns) == final_feats


# Values outside the model's observed range are rejected.
def test_out_of_range_customer_input_is_rejected(customer_values):
    customer_values["Age"] = 100
    with pytest.raises(ValueError, match="Age"):
        prepare_customer_input(customer_values)


# Missing form values fail with a field-specific validation error.
def test_missing_customer_input_is_rejected(customer_values):
    del customer_values["Spending_Score"]
    with pytest.raises(ValueError, match="Spending_Score"):
        prepare_customer_input(customer_values)


# --- clustering ------------------------------------------------------------

# Both app tables report the notebook's complete k=3 through k=8 range.
def test_candidate_tables_match_app_contract(results):
    for metrics in (
        results["two_feature_metrics"],
        results["final_metrics"],
    ):
        assert list(metrics.columns) == ["Clusters", "Inertia", "Silhouette"]
        assert metrics["Clusters"].tolist() == [3, 4, 5, 6, 7, 8]
        assert metrics[["Inertia", "Silhouette"]].notna().all().all()


# Fixed-seed results preserve the notebook's two conclusions.
def test_selected_cluster_counts_match_source_workflow(results):
    assert results["two_feature_k"] == 5
    assert results["selected_k"] == 6


# Selection is calculated from the table rather than hard-coded to six.
def test_selection_rule_uses_highest_silhouette():
    metrics = pd.DataFrame({
        "Clusters": [3, 4, 5],
        "Inertia": [100.0, 80.0, 60.0],
        "Silhouette": [0.2, 0.8, 0.4],
    })
    assert select_cluster_count(metrics) == 4


# Every fit pins the source exercise's initialization and a stable seed.
def test_kmeans_configuration_is_reproducible(df):
    model = fit_kmeans(get_two_feature_data(df), 5)
    assert model.init == "k-means++"
    assert model.n_init == 10
    assert model.random_state == 42


# Profile counts and assignment rows describe the same fitted customers.
def test_cluster_profiles_match_assignments(df, results):
    assignments = results["assignments"]
    profiles = results["profiles"]
    assert len(assignments) == len(df)
    assert assignments["Cluster"].nunique() == results["selected_k"]
    assert profiles["Customers"].sum() == len(df)
    assert profiles["Share"].sum() == pytest.approx(1.0)


# Full app path: three raw values become one valid fitted cluster.
def test_customer_assignment_runs_end_to_end(results, customer_values):
    customer = prepare_customer_input(customer_values)
    cluster = assign_customer(results["model"], results["scaler"], customer)
    assert cluster in set(results["profiles"]["Cluster"])


# Re-running the complete workflow preserves both selections and assignments.
def test_workflow_is_reproducible(df, results):
    repeated = train_clustering_workflow(df)
    assert repeated["two_feature_k"] == results["two_feature_k"]
    assert repeated["selected_k"] == results["selected_k"]
    assert np.array_equal(
        repeated["assignments"]["Cluster"].to_numpy(),
        results["assignments"]["Cluster"].to_numpy(),
    )
