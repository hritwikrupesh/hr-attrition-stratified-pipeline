import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import (
    prepare_features,
    build_preprocessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

ROOT_DIR = Path(__file__).resolve().parent.parent

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODEL_PATH = ROOT_DIR / "models" / "hr_attrition_pipeline.joblib"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER
# ============================================================

def save_figure(filename):
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Created: {path}")


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

X, y, numerical_columns, categorical_columns = (
    prepare_features(df)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

final_pipeline = joblib.load(MODEL_PATH)


# ============================================================
# FIGURE 1
# ATTRITION CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition",
)

plt.title("Employee Attrition Class Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

save_figure(
    "01_attrition_distribution.png"
)


# ============================================================
# FIGURE 2
# DATA QUALITY
# ============================================================

missing_values = df.isnull().sum()

missing_values = (
    missing_values
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 6))

plt.bar(
    missing_values.index,
    missing_values.values,
)

plt.title("Missing Values by Feature")
plt.xlabel("Feature")
plt.ylabel("Missing Values")

plt.xticks(
    rotation=75,
    ha="right",
)

save_figure(
    "02_data_quality.png"
)


# ============================================================
# FIGURE 3
# NUMERICAL DISTRIBUTIONS
# ============================================================

selected_numeric = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "DistanceFromHome",
    "JobLevel",
]

available_numeric = [
    column
    for column in selected_numeric
    if column in df.columns
]

for column in available_numeric:

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True,
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    safe_name = column.lower()

    save_figure(
        f"03_{safe_name}_distribution.png"
    )


# ============================================================
# FIGURE 4
# CATEGORICAL DISTRIBUTIONS
# ============================================================

selected_categorical = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

for column in selected_categorical:

    if column not in df.columns:
        continue

    plt.figure(figsize=(9, 5))

    order = (
        df[column]
        .value_counts()
        .index
    )

    sns.countplot(
        data=df,
        x=column,
        order=order,
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Number of Employees")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    save_figure(
        f"04_{column.lower()}_distribution.png"
    )


# ============================================================
# FIGURE 5
# ATTRITION RATE BY IMPORTANT CATEGORICAL FEATURES
# ============================================================

selected_rate_features = [
    "OverTime",
    "JobRole",
    "Department",
    "BusinessTravel",
    "MaritalStatus",
]

for column in selected_rate_features:

    if column not in df.columns:
        continue

    rate_df = (
        df.groupby(column)["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        rate_df.index.astype(str),
        rate_df.values,
    )

    plt.title(
        f"Attrition Rate by {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Attrition Rate (%)")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    save_figure(
        f"05_{column.lower()}_attrition_rate.png"
    )


# ============================================================
# FIGURE 6
# CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
).copy()

numeric_df["Attrition"] = y.values

correlation_matrix = (
    numeric_df.corr()
)

plt.figure(figsize=(15, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
)

plt.title(
    "Numerical Feature Correlation Heatmap"
)

save_figure(
    "06_correlation_heatmap.png"
)


# ============================================================
# FIGURE 7
# PIPELINE ARCHITECTURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 7)
)

ax.axis("off")

boxes = [
    (
        0.05,
        0.45,
        "Raw HR Dataset",
    ),
    (
        0.25,
        0.65,
        "Numerical Features\nStandardScaler",
    ),
    (
        0.25,
        0.25,
        "Categorical Features\nOneHotEncoder",
    ),
    (
        0.50,
        0.45,
        "ColumnTransformer",
    ),
    (
        0.72,
        0.45,
        "Logistic Regression",
    ),
    (
        0.90,
        0.45,
        "Attrition\nProbability",
    ),
]

for x, y_pos, text in boxes:

    ax.text(
        x,
        y_pos,
        text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="black",
        ),
        transform=ax.transAxes,
    )

ax.annotate(
    "",
    xy=(0.22, 0.53),
    xytext=(0.10, 0.53),
    arrowprops=dict(
        arrowstyle="->"
    ),
    xycoords=ax.transAxes,
)

ax.annotate(
    "",
    xy=(0.48, 0.53),
    xytext=(0.33, 0.67),
    arrowprops=dict(
        arrowstyle="->"
    ),
    xycoords=ax.transAxes,
)

ax.annotate(
    "",
    xy=(0.48, 0.53),
    xytext=(0.33, 0.33),
    arrowprops=dict(
        arrowstyle="->"
    ),
    xycoords=ax.transAxes,
)

ax.annotate(
    "",
    xy=(0.69, 0.53),
    xytext=(0.53, 0.53),
    arrowprops=dict(
        arrowstyle="->"
    ),
    xycoords=ax.transAxes,
)

ax.annotate(
    "",
    xy=(0.87, 0.53),
    xytext=(0.75, 0.53),
    arrowprops=dict(
        arrowstyle="->"
    ),
    xycoords=ax.transAxes,
)

plt.title(
    "HR Attrition Machine Learning Pipeline",
    fontsize=16,
)

save_figure(
    "07_pipeline_architecture.png"
)


# ============================================================
# FIGURE 8
# MODEL PERFORMANCE COMPARISON
# ============================================================

cv_results_path = (
    ROOT_DIR
    / "reports"
    / "cv_results.csv"
)

cv_results = pd.read_csv(
    cv_results_path
)

pivot = cv_results.pivot(
    index="Model",
    columns="Metric",
    values="Mean",
)

metrics_to_plot = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

available_metrics = [
    metric
    for metric in metrics_to_plot
    if metric in pivot.columns
]

plot_data = pivot[
    available_metrics
]

plot_data.plot(
    kind="bar",
    figsize=(12, 7),
)

plt.title(
    "Cross-Validated Model Performance Comparison"
)

plt.xlabel("Model")
plt.ylabel("Score")

plt.xticks(
    rotation=20,
    ha="right",
)

plt.legend(
    title="Metric"
)

save_figure(
    "08_model_comparison.png"
)


# ============================================================
# FIGURE 9
# ROC CURVES
# ============================================================

models_for_roc = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
}

plt.figure(figsize=(8, 6))

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

for model_name, model in models_for_roc.items():

    fold_fpr = []
    fold_tpr = []

    for train_idx, val_idx in cv.split(
        X_train,
        y_train,
    ):

        preprocessor = build_preprocessor(
            numerical_columns,
            categorical_columns,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    clone(model),
                ),
            ]
        )

        pipeline.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        probabilities = pipeline.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_train.iloc[val_idx],
            probabilities,
        )

        fold_fpr.append(fpr)
        fold_tpr.append(tpr)

    mean_fpr = np.linspace(
        0,
        1,
        100,
    )

    interpolated_tpr = []

    for fpr, tpr in zip(
        fold_fpr,
        fold_tpr,
    ):

        interpolated_tpr.append(
            np.interp(
                mean_fpr,
                fpr,
                tpr,
            )
        )

    mean_tpr = np.mean(
        interpolated_tpr,
        axis=0,
    )

    mean_auc = auc(
        mean_fpr,
        mean_tpr,
    )

    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"{model_name} (AUC={mean_auc:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random",
)

plt.title(
    "Cross-Validated ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

save_figure(
    "09_roc_curve.png"
)


# ============================================================
# FIGURE 10
# PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for train_idx, val_idx in cv.split(
    X_train,
    y_train,
):

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx],
    )

    probabilities = pipeline.predict_proba(
        X_train.iloc[val_idx]
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_train.iloc[val_idx],
        probabilities,
    )

    plt.plot(
        recall,
        precision,
        alpha=0.7,
    )

plt.axhline(
    y=y_train.mean(),
    linestyle="--",
    label="Positive Class Baseline",
)

plt.title(
    "Cross-Validated Precision-Recall Curves"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.legend()

save_figure(
    "10_precision_recall_curves.png"
)


# ============================================================
# FIGURE 11
# CONFUSION MATRIX
# ============================================================

y_test_pred = final_pipeline.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_test_pred,
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Attrition",
        "Attrition",
    ],
    yticklabels=[
        "No Attrition",
        "Attrition",
    ],
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

save_figure(
    "11_confusion_matrix.png"
)


# ============================================================
# FIGURE 12
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

permutation = permutation_importance(
    final_pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": permutation.importances_mean,
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False,
    )
    .head(15)
)

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"][::-1],
    feature_importance["Importance"][::-1],
)

plt.title(
    "Permutation Feature Importance - Logistic Regression"
)

plt.xlabel("Mean Decrease in ROC-AUC")
plt.ylabel("Feature")

save_figure(
    "12_permutation_feature_importance.png"
)


# ============================================================
# FIGURE 13
# CV SCORE DISTRIBUTION
# ============================================================

fold_scores = []

for train_idx, val_idx in cv.split(
    X_train,
    y_train,
):

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx],
    )

    probabilities = pipeline.predict_proba(
        X_train.iloc[val_idx]
    )[:, 1]

    score = average_precision_score(
        y_train.iloc[val_idx],
        probabilities,
    )

    fold_scores.append(score)

plt.figure(figsize=(8, 6))

plt.boxplot(
    fold_scores,
)

plt.title(
    "5-Fold Cross-Validation PR-AUC Distribution"
)

plt.ylabel("PR-AUC")

save_figure(
    "13_cv_score_distribution.png"
)


# ============================================================
# FIGURE 14
# LEARNING CURVE
# ============================================================

learning_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            build_preprocessor(
                numerical_columns,
                categorical_columns,
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

train_sizes, train_scores, validation_scores = learning_curve(
    learning_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    train_sizes=np.linspace(
        0.2,
        1.0,
        5,
    ),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

plt.figure(figsize=(8, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training ROC-AUC",
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation ROC-AUC",
)

plt.title(
    "Learning Curve - Logistic Regression"
)

plt.xlabel("Training Examples")
plt.ylabel("ROC-AUC")

plt.legend()

save_figure(
    "14_learning_curve.png"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)

print("\nFigures saved to:")
print(FIGURES_DIR)