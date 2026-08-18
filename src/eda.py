from pathlib import Path
import sys

import pandas as pd
import numpy as np


# Allow importing data_loader.py when running this file directly
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data


def main():
    df = load_data()

    print("\n" + "=" * 70)
    print("HR ATTRITION DATASET - EDA")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Dataset shape
    # ---------------------------------------------------------
    print("\n1. DATASET SHAPE")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # ---------------------------------------------------------
    # 2. Column information
    # ---------------------------------------------------------
    print("\n2. COLUMN INFORMATION")
    print("-" * 70)

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum()
    })

    print(info.to_string(index=False))

    # ---------------------------------------------------------
    # 3. Numerical and categorical columns
    # ---------------------------------------------------------
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\n3. FEATURE TYPES")
    print("-" * 70)

    print(f"Numerical features   : {len(numerical_columns)}")
    print(f"Categorical features : {len(categorical_columns)}")

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # ---------------------------------------------------------
    # 4. Target analysis
    # ---------------------------------------------------------
    target = "Attrition"

    if target not in df.columns:
        raise ValueError(
            f"Expected target column '{target}' was not found."
        )

    print("\n4. TARGET DISTRIBUTION")
    print("-" * 70)

    target_counts = df[target].value_counts()
    target_percentages = df[target].value_counts(normalize=True) * 100

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percentages.round(2)
    })

    print(target_summary)

    # ---------------------------------------------------------
    # 5. Missing values
    # ---------------------------------------------------------
    print("\n5. MISSING VALUES")
    print("-" * 70)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    # ---------------------------------------------------------
    # 6. Duplicate rows
    # ---------------------------------------------------------
    print("\n6. DUPLICATES")
    print("-" * 70)

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    # ---------------------------------------------------------
    # 7. Constant columns
    # ---------------------------------------------------------
    print("\n7. CONSTANT COLUMNS")
    print("-" * 70)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        print(constant_columns)
    else:
        print("No constant columns found.")

    # ---------------------------------------------------------
    # 8. Identifier-like columns
    # ---------------------------------------------------------
    print("\n8. POTENTIAL IDENTIFIER COLUMNS")
    print("-" * 70)

    identifier_candidates = []

    for column in df.columns:
        unique_ratio = df[column].nunique(dropna=False) / len(df)

        if unique_ratio >= 0.95:
            identifier_candidates.append(
                (column, round(unique_ratio, 4))
            )

    if identifier_candidates:
        for column, ratio in identifier_candidates:
            print(f"{column}: unique ratio = {ratio}")
    else:
        print("No obvious identifier-like columns detected.")

    # ---------------------------------------------------------
    # 9. Numerical descriptive statistics
    # ---------------------------------------------------------
    print("\n9. NUMERICAL DESCRIPTIVE STATISTICS")
    print("-" * 70)

    print(
        df[numerical_columns]
        .describe()
        .round(2)
        .to_string()
    )

    # ---------------------------------------------------------
    # 10. Categorical cardinality
    # ---------------------------------------------------------
    print("\n10. CATEGORICAL CARDINALITY")
    print("-" * 70)

    categorical_summary = pd.DataFrame({
        "Column": categorical_columns,
        "Unique Values": [
            df[column].nunique()
            for column in categorical_columns
        ]
    })

    print(categorical_summary.to_string(index=False))

    # ---------------------------------------------------------
    # 11. Numerical correlations with target
    # ---------------------------------------------------------
    print("\n11. NUMERICAL CORRELATIONS")
    print("-" * 70)

    encoded_target = df[target].map({
        "Yes": 1,
        "No": 0
    })

    if encoded_target.notna().all():
        correlation_df = df[numerical_columns].copy()
        correlation_df["Attrition"] = encoded_target

        correlations = (
            correlation_df
            .corr(numeric_only=True)["Attrition"]
            .drop("Attrition")
            .sort_values(
                key=lambda x: x.abs(),
                ascending=False
            )
        )

        print(correlations.round(4))

    # ---------------------------------------------------------
    # 12. Important categorical attrition rates
    # ---------------------------------------------------------
    print("\n12. CATEGORICAL ATTRITION RATES")
    print("-" * 70)

    selected_categories = [
        "OverTime",
        "JobRole",
        "Department",
        "BusinessTravel",
        "MaritalStatus",
        "JobSatisfaction"
    ]

    for column in selected_categories:

        if column not in df.columns:
            continue

        print(f"\n{column}")

        if df[column].dtype == "object":
            rate = (
                df.groupby(column)[target]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            print(rate.round(2).to_string())

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()