import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import prepare_features


RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    df = load_data()

    X, y, numerical_columns, categorical_columns = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("=" * 70)
    print("STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 70)

    print(f"\nTotal samples : {len(X)}")
    print(f"Training      : {len(X_train)}")
    print(f"Testing       : {len(X_test)}")

    print("\nOriginal target distribution:")
    print(
        y.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTraining target distribution:")
    print(
        y_train.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTesting target distribution:")
    print(
        y_test.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class counts:")
    print(y_test.value_counts().sort_index())

    print("\nStratification test completed successfully!")


if __name__ == "__main__":
    main()