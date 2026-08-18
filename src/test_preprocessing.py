import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import prepare_features, build_preprocessor


def main():

    df = load_data()

    X, y, numerical_columns, categorical_columns = prepare_features(df)

    print("=" * 70)
    print("PREPROCESSING TEST")
    print("=" * 70)

    print(f"\nOriginal dataset shape : {df.shape}")
    print(f"Feature matrix shape   : {X.shape}")
    print(f"Target shape           : {y.shape}")

    print("\nTarget values:")
    print(y.value_counts())

    print("\nNumerical columns:")
    print(f"Count: {len(numerical_columns)}")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(f"Count: {len(categorical_columns)}")
    print(categorical_columns)

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    transformed = preprocessor.fit_transform(X)

    print("\nTransformed feature matrix:")
    print(f"Shape: {transformed.shape}")

    print("\nPreprocessing test completed successfully!")


if __name__ == "__main__":
    main()