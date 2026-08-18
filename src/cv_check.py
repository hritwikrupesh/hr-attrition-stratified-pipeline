import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import prepare_features


RANDOM_STATE = 42
N_SPLITS = 5


def main():

    df = load_data()

    X, y, _, _ = prepare_features(df)

    # First create the untouched test partition.
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Number of folds  : {N_SPLITS}")

    print("\nFold distributions:")
    print("-" * 70)

    for fold, (train_idx, validation_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):

        y_fold_train = y_train.iloc[train_idx]
        y_fold_validation = y_train.iloc[validation_idx]

        train_positive_rate = y_fold_train.mean() * 100
        validation_positive_rate = (
            y_fold_validation.mean() * 100
        )

        print(
            f"Fold {fold}: "
            f"train={len(train_idx):4d}, "
            f"validation={len(validation_idx):4d}, "
            f"train attrition={train_positive_rate:5.2f}%, "
            f"validation attrition={validation_positive_rate:5.2f}%"
        )

    print("\nStratified 5-fold validation setup is correct.")


if __name__ == "__main__":
    main()