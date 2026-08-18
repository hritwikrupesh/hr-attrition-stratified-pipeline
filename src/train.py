import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline


sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import prepare_features, build_preprocessor


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def build_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true, y_pred, y_probability):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }


def main():

    print("=" * 80)
    print("HR ATTRITION - STRATIFIED CROSS-VALIDATED ML PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load and prepare dataset
    # ---------------------------------------------------------

    df = load_data()

    X, y, numerical_columns, categorical_columns = prepare_features(df)

    # ---------------------------------------------------------
    # 2. Stratified train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nDataset:")
    print(f"Total samples : {len(X)}")
    print(f"Training      : {len(X_train)}")
    print(f"Testing       : {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Stratified K-Fold
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results = []

    # ---------------------------------------------------------
    # 4. Cross-validation
    # ---------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train),
            start=1,
        ):

            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # IMPORTANT:
            # A fresh preprocessing pipeline is created
            # for every fold to prevent data leakage.
            preprocessor = build_preprocessor(
                numerical_columns,
                categorical_columns,
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", clone(model)),
                ]
            )

            # Fit ONLY on the current training fold.
            pipeline.fit(
                X_fold_train,
                y_fold_train,
            )

            # Predict validation fold.
            y_pred = pipeline.predict(X_fold_val)

            y_probability = pipeline.predict_proba(
                X_fold_val
            )[:, 1]

            metrics = calculate_metrics(
                y_fold_val,
                y_pred,
                y_probability,
            )

            fold_metrics.append(metrics)

            print(
                f"Fold {fold}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

        # -----------------------------------------------------
        # Mean and standard deviation
        # -----------------------------------------------------

        print("\nCross-validation summary:")

        for metric_name in fold_metrics[0]:

            values = [
                fold[metric_name]
                for fold in fold_metrics
            ]

            mean_value = np.mean(values)
            std_value = np.std(
                values,
                ddof=1,
            )

            print(
                f"{metric_name.upper():10}: "
                f"{mean_value:.4f} ± {std_value:.4f}"
            )

            results.append({
                "Model": model_name,
                "Metric": metric_name,
                "Mean": mean_value,
                "Std": std_value,
            })

    # ---------------------------------------------------------
    # 5. Results table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "cv_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()