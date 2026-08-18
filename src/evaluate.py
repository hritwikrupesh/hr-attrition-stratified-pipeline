import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import (
    prepare_features,
    build_preprocessor,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    print("=" * 80)
    print("FINAL MODEL TRAINING AND HELD-OUT TEST EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = load_data()

    X, y, numerical_columns, categorical_columns = (
        prepare_features(df)
    )

    # ---------------------------------------------------------
    # 2. Same stratified split used during CV
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nData:")
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Build selected model
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ---------------------------------------------------------
    # 4. Fit on COMPLETE training partition
    # ---------------------------------------------------------

    print("\nTraining final Logistic Regression model...")

    final_pipeline.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # 5. Final prediction on untouched test set
    # ---------------------------------------------------------

    y_pred = final_pipeline.predict(X_test)

    y_probability = final_pipeline.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # 6. Calculate final metrics
    # ---------------------------------------------------------

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_probability,
        ),

        "PR-AUC": average_precision_score(
            y_test,
            y_probability,
        ),
    }

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    for metric, value in metrics.items():
        print(f"{metric:12}: {value:.4f}")

    # ---------------------------------------------------------
    # 8. Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Attrition",
                "Attrition",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 9. Save test results
    # ---------------------------------------------------------

    reports_dir = (
        Path(__file__).resolve().parent.parent
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
            }
        ]
    )

    results_path = reports_dir / "test_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Save confusion matrix
    # ---------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Attrition",
            "Actual Attrition",
        ],
        columns=[
            "Predicted No Attrition",
            "Predicted Attrition",
        ],
    )

    cm_path = reports_dir / "confusion_matrix.csv"

    cm_df.to_csv(cm_path)

    # ---------------------------------------------------------
    # 11. Save final model
    # ---------------------------------------------------------

    models_dir = (
        Path(__file__).resolve().parent.parent
        / "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / "hr_attrition_pipeline.joblib"
    )

    metadata_path = (
        models_dir
        / "pipeline_meta.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    joblib.dump(
        {
            "features": list(X.columns),
            "target": "Attrition",
            "model": "Logistic Regression",
            "random_state": RANDOM_STATE,
        },
        metadata_path,
    )

    print("\n" + "=" * 80)
    print("ARTIFACTS SAVED")
    print("=" * 80)

    print(f"Test results     : {results_path}")
    print(f"Confusion matrix : {cm_path}")
    print(f"Model            : {model_path}")
    print(f"Metadata         : {metadata_path}")

    print("\nFinal evaluation completed successfully.")


if __name__ == "__main__":
    main()