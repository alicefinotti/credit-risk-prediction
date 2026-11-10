# Credit Risk Prediction

**Python | Machine Learning | Classification**

## Overview

University group project investigating the prediction of serious credit delinquency (90+ days past due) within a two-year horizon. The analysis uses **112,500 training observations** before cleaning, with approximately **6.7% positive cases**, making class imbalance a central modeling challenge.

## Methodology

- **Data preprocessing:** Exploratory analysis, outlier treatment, and comparison of mean, median, KNN, and Random Forest-based iterative imputation for missing income values.
- **Feature engineering and selection:** Financial ratios and polynomial features, followed by recursive feature elimination with cross-validation (RFECV) and correlation analysis to reduce the feature set.
- **Model comparison:** Logistic Regression, bagged decision trees, XGBoost, and Histogram-based Gradient Boosting, with hyperparameter tuning and evaluation using precision, recall, F1 score, and ROC AUC.

## Key Results

**XGBoost** was selected based on its balance between identifying potential defaults and limiting false-positive predictions. In the original evaluation, the tuned model achieved **72.6% recall** and **18.0% precision**, illustrating the trade-off involved in identifying relatively rare events.

## Files

- [`credit_risk_prediction.ipynb`](credit_risk_prediction.ipynb) — Full analysis, modeling code, and saved visualizations.
- [`report.pdf`](report.pdf) — Detailed methodology, model comparison, and conclusions.

## Data Availability

The notebook expects `train.csv` and `test.csv` in the project directory. These original project datasets are **not included** in this repository. The saved notebook outputs can be reviewed without them, but rerunning the analysis requires the original data and the Python packages used in the notebook (including pandas, NumPy, SciPy, scikit-learn, XGBoost, Matplotlib, and Seaborn).
