from typing import Tuple, List

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Attrition"

# Columns identified during EDA as non-predictive / identifiers.
COLUMNS_TO_DROP = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate target and predictors and remove non-predictive columns.

    Returns:
        X: Feature dataframe
        y: Target series
        numerical_columns: Numerical feature names
        categorical_columns: Categorical feature names
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # Remove columns that should not be used as predictors.
    drop_columns = [
        column
        for column in COLUMNS_TO_DROP
        if column in df.columns
    ]

    X = df.drop(
        columns=[TARGET_COLUMN] + drop_columns
    ).copy()

    y = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    if y.isnull().any():
        raise ValueError(
            "Unexpected values found in Attrition target."
        )

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    return (
        X,
        y,
        numerical_columns,
        categorical_columns,
    )


def build_preprocessor(
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Build leakage-safe preprocessing.

    Numerical:
        StandardScaler

    Categorical:
        OneHotEncoder(handle_unknown='ignore')
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor