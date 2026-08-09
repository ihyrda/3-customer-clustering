"""Dataset loading, identity validation, and app summary metrics.

load_data - read the bundled course CSV
validate_data - confirm the expected file structure
get_data_summary - return metrics used by the overview tab
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parents[1]
d_path = root / "data" / "mall_customers.csv"

id = "Customer_ID"
desc = "Gender"
final_feats = ["Age", "Annual_Income", "Spending_Score"]
two_feats = ["Annual_Income", "Spending_Score"]
required_cols = [id, desc] + final_feats


def load_data(path=None) -> pd.DataFrame:
    """Read the course dataset from disk; handle file-level failures only."""
    path = Path(path) if path is not None else d_path
    logger.info("Loading dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Mall customer dataset not found at {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at {path} is empty or unreadable.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at {path} is malformed: {exc}") from exc

    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Confirm the loaded frame is the intended course dataset.

    The CSV is a fixed project asset, so validation checks identity rather
    than policing values that cannot change independently of the file.
    """
    if df.shape != (200, 5):
        raise ValueError(f"Expected dataset shape (200, 5); received {df.shape}.")

    if list(df.columns) != required_cols:
        raise ValueError(f"Expected columns in this order: {required_cols}")

    numeric_columns = [id] + final_feats
    non_numeric = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Expected numeric columns are non-numeric: {non_numeric}")

    logger.info("Dataset identity confirmed")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return the dataset metrics displayed on the overview tab."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "gender_counts": df[desc].value_counts().to_dict(),
    }
