"""Feature selection, scaling, and preparation of one customer row.

get_two_feature_data - reproduce the notebook's raw teaching view
get_three_feature_data - return the final feature view in frozen order
scale_features - fit and apply the final StandardScaler
prepare_customer_input - validate and order app input
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import final_feats, two_feats

logger = logging.getLogger(__name__)

# App bounds match the data seen by the model; K-means always assigns a
# cluster, so accepting farther values would imply unsupported coverage.
input_ranges = {
    "Age": (18, 70),
    "Annual_Income": (15, 137),
    "Spending_Score": (1, 99),
}


def get_two_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the notebook's income-and-spending view on raw units."""
    return df[two_feats].copy()


def get_three_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the final three numeric features in frozen order."""
    return df[final_feats].copy()


def scale_features(X: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    """Fit one scaler and transform the final feature view."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    logger.info("Scaled %d rows across %d features", *X.shape)
    return scaler, scaled


def prepare_customer_input(values: dict) -> pd.DataFrame:
    """Create one validated customer row in final-model feature order."""
    unknown = sorted(set(values) - set(final_feats))
    if unknown:
        raise ValueError(f"Unknown input fields: {unknown}")

    missing = [column for column in final_feats if values.get(column) is None]
    if missing:
        raise ValueError(f"Missing required customer inputs: {missing}")

    converted = {}
    for column, (low, high) in input_ranges.items():
        try:
            value = float(values[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{column}' must be numeric.") from exc
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"'{column}' must fall between {low} and {high}.")
        converted[column] = value

    row = pd.DataFrame([converted], columns=final_feats)
    logger.info("Prepared one customer row")
    return row
