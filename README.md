# HR Attrition Stratified Pipeline

## Interactive Employee Attrition Prediction & Analytics

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> An end-to-end machine-learning pipeline for employee attrition prediction featuring leakage-safe preprocessing, stratified validation, model evaluation, permutation feature-importance analysis, model persistence, and an interactive Streamlit prototype.

---

## 🚀 Live Prototype

### [HR Pulse — Attrition Intelligence](https://hr-attrition-stratified-pipeline.streamlit.app/)

The deployed application provides an interactive interface for:

- Employee profile assessment
- Attrition-risk prediction
- Model-estimated probability
- Risk classification
- Prediction-driver analysis
- Model intelligence
- Evaluation visualizations
- ML pipeline visualization
- Responsible-use communication

---

## 🔗 Project Resources

| Resource | Link |
|---|---|
| 🌐 Live link | [Open Application](https://hr-attrition-stratified-pipeline.streamlit.app/) |
| 💻 GitHub Repository | [View Repository](https://github.com/hritwikrupesh/hr-attrition-stratified-pipeline) |

---

# 📌 Project Overview

Employee attrition is an important organizational challenge because unexpected employee turnover can affect workforce planning, productivity, operational continuity, and recruitment effort.

This project implements a complete machine-learning workflow for predicting employee attrition using structured HR employee data.

The system combines:

1. Data loading and validation
2. Exploratory data analysis
3. Leakage-safe preprocessing
4. Numerical feature standardization
5. Categorical one-hot encoding
6. Stratified train/test splitting
7. Stratified 5-fold cross-validation
8. Multiple classification models
9. Held-out test evaluation
10. Permutation feature-importance analysis
11. Model artifact generation
12. Interactive Streamlit deployment

The resulting machine-learning pipeline is integrated into an enterprise-style HR analytics prototype called:

> **HR Pulse — Attrition Intelligence**

---

# 🎯 Objectives

The primary objectives of the project are to:

- Build a reproducible employee attrition prediction pipeline.
- Preserve class proportions through stratified sampling.
- Reduce preprocessing leakage during model validation.
- Compare multiple machine-learning classifiers.
- Evaluate models using multiple classification metrics.
- Analyze model-level feature importance.
- Persist the trained model pipeline for inference.
- Provide an interactive employee assessment interface.
- Present prediction results through an enterprise-style dashboard.
- Provide model intelligence and diagnostic visualizations.
- Communicate responsible-use considerations for HR analytics.

---

# 🧠 End-to-End Machine Learning Workflow

```text
                    HR Employee Dataset
                           │
                           ▼
                 Data Validation & EDA
                           │
                           ▼
              Stratified Train/Test Split
                           │
                           ▼
                 Preprocessing Pipeline
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Numerical Features    Categorical Features
                 │                   │
                 ▼                   ▼
            StandardScaler      OneHotEncoder
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    ColumnTransformer
                           │
                           ▼
                 Machine Learning Models
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Logistic       Random       Gradient
        Regression       Forest       Boosting
              │            │            │
              └────────────┼────────────┘
                           ▼
                  Stratified 5-Fold CV
                           │
                           ▼
                     Model Selection
                           │
                           ▼
                 Held-Out Test Evaluation
                           │
                           ▼
                Permutation Importance
                           │
                           ▼
                 Persisted ML Pipeline
                           │
                           ▼
                  Streamlit Prototype
                           │
                           ▼
                    Cloud Deployment
```

---

# 📊 Dataset

The project uses the IBM HR Employee Attrition dataset.

The dataset contains:

- **1,470 employee records**
- **35 original attributes**
- Demographic information
- Job and organizational information
- Compensation information
- Career history
- Satisfaction and wellbeing indicators
- `Attrition` as the prediction target

## Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| No Attrition | 1,233 | 83.88% |
| Attrition | 237 | 16.12% |
| **Total** | **1,470** | **100%** |

The target is imbalanced, making stratified validation and metrics beyond accuracy particularly important.

---

# 🔧 Data Preprocessing

The preprocessing workflow separates numerical and categorical variables.

## Numerical Features

Numerical features are standardized using:

```text
StandardScaler
```

## Categorical Features

Categorical variables are transformed using:

```text
OneHotEncoder
```

## Combined Transformation

The numerical and categorical preprocessing branches are combined using:

```text
ColumnTransformer
```

The preprocessing pipeline is fitted only on the relevant training data during cross-validation, reducing the risk of data leakage.

---

# ⚖️ Stratified Train/Test Split

The dataset is divided using a stratified train/test split.

```text
Total samples     : 1,470
Training samples  : 1,176
Testing samples   : 294
```

The stratification process maintains a similar class distribution between the training and testing datasets.

## Original Distribution

```text
No Attrition : 83.88%
Attrition    : 16.12%
```

## Training Distribution

```text
No Attrition : 83.84%
Attrition    : 16.16%
```

## Testing Distribution

```text
No Attrition : 84.01%
Attrition    : 15.99%
```

---

# 🔬 Stratified 5-Fold Cross-Validation

The training dataset is evaluated using:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

The stratified approach preserves the minority-class proportion across the validation folds.

This is especially important because employee attrition represents a minority class in the dataset.

The implementation also rebuilds the preprocessing pipeline inside each validation fold so that transformations are learned only from the corresponding training portion.

---

# 🤖 Machine Learning Models

The project evaluates multiple classification models.

## 1. Logistic Regression

Logistic Regression provides a strong and relatively interpretable baseline for binary classification.

It is also used as the final deployed model in the current prototype.

## 2. Random Forest

Random Forest is an ensemble tree-based classifier capable of modelling nonlinear relationships and feature interactions.

## 3. Gradient Boosting

Gradient Boosting builds an ensemble sequentially, with later models focusing on improving errors made by previous models.

## 4. Balanced Logistic Regression

A balanced Logistic Regression configuration is also evaluated to examine the effect of class weighting on minority-class recall.

---

# 📈 Cross-Validation Results

The current implementation evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

## Logistic Regression — 5-Fold Cross-Validation

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.8912 | 0.0169 |
| Precision | 0.7927 | 0.0860 |
| Recall | 0.4421 | 0.0798 |
| F1 | 0.5653 | 0.0799 |
| ROC-AUC | 0.8379 | 0.0339 |
| PR-AUC | 0.6518 | 0.0691 |

The complete cross-validation results are stored in:

```text
reports/cv_results.csv
```

---

# 🧪 Held-Out Test Evaluation

The final Logistic Regression pipeline is evaluated on the held-out test set.

```text
Training samples : 1,176
Test samples     : 294
```

## Test Results

| Metric | Result |
|---|---:|
| Accuracy | 86.39% |
| Precision | 64.00% |
| Recall | 34.04% |
| F1 Score | 44.44% |
| ROC-AUC | 0.8102 |
| PR-AUC | 0.5927 |

## Confusion Matrix

```text
                    Predicted
                 No Attrition  Attrition

Actual
No Attrition          238          9
Attrition              31         16
```

The complete test results are available in:

```text
reports/test_results.csv
reports/confusion_matrix.csv
```

---

# 📊 Model Evaluation

The project generates multiple evaluation visualizations:

- Model comparison
- ROC curves
- Precision-Recall curves
- Confusion matrix
- Cross-validation score distribution
- Learning curve

These visualizations provide a broader understanding of classifier behaviour instead of relying solely on accuracy.

---

# 🔍 Prediction Drivers & Model Interpretability

The prototype includes a dedicated **Prediction Drivers** section.

Permutation feature importance is used to identify model-level signals by measuring how predictive performance changes when individual features are shuffled.

The documented top signals include:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | OverTime | 0.1072 |
| 2 | NumCompaniesWorked | 0.0256 |
| 3 | MaritalStatus | 0.0232 |
| 4 | YearsSinceLastPromotion | 0.0231 |
| 5 | BusinessTravel | 0.0212 |

### Interpretation

The feature-importance values represent **model-level predictive sensitivity**.

They should not be interpreted as proof that a feature causally causes an employee to leave.

The complete visualization is available at:

```text
reports/figures/12_permutation_feature_importance.png
```

---

# 🖥️ HR Pulse — Attrition Intelligence

The machine-learning pipeline is integrated into an interactive Streamlit application.

The application provides an enterprise-oriented interface for exploring employee attrition risk.

## Employee Assessment

Users can enter structured employee information covering:

- Personal Profile
- Job Profile
- Compensation
- Career History
- Performance
- Wellbeing

## Attrition Risk Prediction

The application generates:

- Predicted attrition class
- Model-estimated probability
- Risk classification
- Probability visualization
- Assessment summary

## Prediction Drivers

The prototype provides a model-intelligence section presenting important model-level signals.

## Model Intelligence

The dashboard provides:

- Cross-validation performance
- Model comparison
- ROC analysis
- Precision-Recall analysis
- Confusion matrix
- Cross-validation stability
- Learning behaviour

---

# 🖼️ Prototype Preview

The prototype follows an enterprise analytics dashboard design with:

- Structured information hierarchy
- Rounded analytical cards
- Clear input/output separation
- Risk-focused visualizations
- Model explainability
- Professional data presentation
- Section-based navigation
- Responsible-use messaging

### Live Application

👉 **[Launch HR Pulse — Attrition Intelligence](https://hr-attrition-stratified-pipeline.streamlit.app/)**

---

# 🏗️ ML Pipeline Architecture

The deployed pipeline follows this structure:

```text
Raw Employee Data
        │
        ▼
Data Validation
        │
        ▼
Feature Preparation
        │
        ├─────────────────────┐
        ▼                     ▼
Numerical Features      Categorical Features
        │                     │
        ▼                     ▼
 StandardScaler          OneHotEncoder
        │                     │
        └──────────┬──────────┘
                   ▼
            ColumnTransformer
                   │
                   ▼
          Logistic Regression
                   │
                   ▼
          Attrition Probability
                   │
                   ▼
            Risk Classification
                   │
                   ▼
         Streamlit Presentation
```

---

# 📁 Repository Structure

```text
hr-attrition-stratified-pipeline/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   ├── hr_attrition_pipeline.joblib
│   └── pipeline_meta.joblib
│
├── notebooks/
│   └── 01_eda_and_visualization.ipynb
│
├── reports/
│   ├── confusion_matrix.csv
│   ├── cv_results.csv
│   ├── test_results.csv
│   │
│   └── figures/
│       ├── 01_attrition_distribution.png
│       ├── 02_data_quality.png
│       ├── 03_age_distribution.png
│       ├── 03_distancefromhome_distribution.png
│       ├── 03_joblevel_distribution.png
│       ├── 03_monthlyincome_distribution.png
│       ├── 03_totalworkingyears_distribution.png
│       ├── 03_yearsatcompany_distribution.png
│       ├── 04_businesstravel_distribution.png
│       ├── 04_department_distribution.png
│       ├── 04_educationfield_distribution.png
│       ├── 04_jobrole_distribution.png
│       ├── 04_maritalstatus_distribution.png
│       ├── 04_overtime_distribution.png
│       ├── 05_businesstravel_attrition_rate.png
│       ├── 05_department_attrition_rate.png
│       ├── 05_jobrole_attrition_rate.png
│       ├── 05_maritalstatus_attrition_rate.png
│       ├── 05_overtime_attrition_rate.png
│       ├── 06_correlation_heatmap.png
│       ├── 07_pipeline_architecture.png
│       ├── 08_model_comparison.png
│       ├── 09_roc_curve.png
│       ├── 10_precision_recall_curves.png
│       ├── 11_confusion_matrix.png
│       ├── 12_permutation_feature_importance.png
│       ├── 13_cv_score_distribution.png
│       └── 14_learning_curve.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── cv_check.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── eda.py
│   ├── generate_figures.py
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/hritwikrupesh/hr-attrition-stratified-pipeline.git
```

```bash
cd hr-attrition-stratified-pipeline
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

From the project root:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

---

# 🧪 Reproduce the ML Pipeline

## Test preprocessing

```bash
python src/test_preprocessing.py
```

## Verify stratified train/test split

```bash
python src/split_data.py
```

## Verify stratified cross-validation

```bash
python src/cv_check.py
```

## Train and compare models

```bash
python src/train.py
```

## Evaluate the final model

```bash
python src/evaluate.py
```

## Generate figures

```bash
python src/generate_figures.py
```

---

# 💾 Model Artifacts

The final trained pipeline is persisted using Joblib:

```text
models/hr_attrition_pipeline.joblib
```

Pipeline metadata is stored in:

```text
models/pipeline_meta.joblib
```

The Streamlit application loads the persisted model artifacts directly rather than retraining the model every time the application starts.

---

# 📊 Generated Analytical Outputs

The project generates the following analytical outputs:

- Attrition distribution
- Data-quality analysis
- Numerical feature distributions
- Categorical feature distributions
- Attrition-rate comparisons
- Correlation heatmap
- Pipeline architecture
- Model comparison
- ROC curves
- Precision-Recall curves
- Confusion matrix
- Permutation feature importance
- Cross-validation score distribution
- Learning curve

All generated figures are stored under:

```text
reports/figures/
```

---

# ☁️ Deployment

The Streamlit prototype is deployed using **Streamlit Community Cloud**.

## Production Prototype

👉 **https://hr-attrition-stratified-pipeline.streamlit.app/**

The deployed application uses the GitHub repository as its source.

### Entry Point

```text
app/streamlit_app.py
```

### Dependencies

```text
requirements.txt
```

### Model Artifacts

```text
models/
```

### Analytical Figures

```text
reports/figures/
```

---

# 🛡️ Responsible Use

This project is intended as an educational and analytical prototype.

Employee attrition predictions are statistical estimates and should not be treated as definitive judgments about individual employees.

The system should **not** be used as the sole basis for:

- Hiring decisions
- Promotion decisions
- Disciplinary actions
- Termination decisions
- Other consequential employment decisions

Human judgment, organizational policies, contextual information, and appropriate governance remain essential.

---

# ⚠️ Limitations

The project has several limitations:

- The model is trained on a specific HR dataset.
- Model performance may not generalize to every organization.
- Historical data may contain relationships that do not represent future workforce behaviour.
- Attrition prediction is probabilistic rather than deterministic.
- Model outputs should be interpreted within organizational context.
- Additional validation would be required before production deployment.
- Model monitoring and drift detection would be required for long-term operational use.

---

# 🔮 Future Enhancements

Potential future improvements include:

- Model monitoring
- Data-drift detection
- Probability calibration
- Decision-threshold optimization
- Additional explainability techniques
- Automated retraining workflows
- Cloud-based production architecture
- Role-based access control
- Audit logging
- Fairness monitoring
- Additional external validation
- Continuous model-performance monitoring

These are future enhancements and are not represented as current prototype capabilities.

---

# 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Numerical Preprocessing | StandardScaler |
| Categorical Preprocessing | OneHotEncoder |
| Feature Transformation | ColumnTransformer |
| Classification Models | Logistic Regression, Random Forest, Gradient Boosting |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Interactive Application | Streamlit |
| Notebook | Jupyter Notebook |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

# 📌 Key Project Highlights

| Metric | Value |
|---|---:|
| Employee Records | 1,470 |
| Original Dataset Attributes | 35 |
| Stratified CV | 5-Fold |
| Models / Configurations Evaluated | 4 |
| Generated Analytical Figures | 14 |
| Held-Out Test Accuracy | 86.39% |
| Held-Out Test ROC-AUC | 0.8102 |
| Deployment | Streamlit Community Cloud |

---

# 🔗 Project Links

### 🌐 Live Prototype

**[HR Pulse — Attrition Intelligence](https://hr-attrition-stratified-pipeline.streamlit.app/)**

### 💻 GitHub Repository

**[hr-attrition-stratified-pipeline](https://github.com/hritwikrupesh/hr-attrition-stratified-pipeline)**

---

# 👨‍💻 Author

**Gollu Hritwik Rupesh**

B.Tech — Computer Science and Engineering

ANITS

---

# 📜 Project Status

**Completed — ML Pipeline + Interactive Prototype + Cloud Deployment**

The project implements an end-to-end employee attrition prediction workflow covering:

```text
Data
  ↓
Preprocessing
  ↓
Stratified Validation
  ↓
Model Training
  ↓
Evaluation
  ↓
Interpretability
  ↓
Model Persistence
  ↓
Interactive Prototype
  ↓
Cloud Deployment
```

---

## ⭐ Acknowledgement

This project was developed as part of an IBM project/case-study initiative focused on applying machine-learning techniques to a practical employee attrition prediction problem.
