import pandas as pd
import pytest

from src.data.validate_data import validate_data, DataValidationError


def make_df(amounts, is_frauds):
    return pd.DataFrame({"amount": amounts, "is_fraud": is_frauds})


def test_missing_columns():
    df = pd.DataFrame({"amount": [10.0]})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(df)


def test_missing_values():
    df = make_df([None, 10.0], [0, 1])
    with pytest.raises(DataValidationError, match="Missing values detected"):
        validate_data(df)


def test_non_numeric_amount():
    df = make_df(["a", "b"], [0, 1])
    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_data(df)


def test_negative_amount():
    df = make_df([-1, 10], [0, 1])
    with pytest.raises(DataValidationError, match="Negative transaction"):
        validate_data(df)


def test_non_binary_is_fraud():
    df = make_df([1, 2, 3], [0, 1, 2])
    with pytest.raises(DataValidationError, match="must be binary"):
        validate_data(df)


def test_valid_data_passes():
    df = make_df([0.0, 10.0, 5.5], [0, 1, 0])
    # Should not raise
    validate_data(df)
