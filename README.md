# Predictive Insurance Model — Proposal

## Project Description

Insurance providers need to estimate claim risk early to price policies fairly and manage loss exposure. The problem is a binary classification task: given a policy's vehicle and policyholder attributes, predict whether a claim will be filed (`is_claim`). The dataset is imbalanced (claims are relatively rare), so the modeling approach must handle skewed classes while still producing actionable signals for underwriting and portfolio monitoring.

This project builds a model for a unified dataset by combining vehicle, policy, and historical claim data, then trains and evaluates multiple classifiers to identify the most effective model for claim prediction.

## Goals
- Predict the likelihood of a claim (`is_claim`) for each policy.
- Compare multiple classification models and select the best performing approach for imbalanced data.
- Balance accuracy with recall/precision to reduce missed claims and false alarms.
- Produce interpretable insights on which vehicle and policy attributes correlate with claims.

## Project Timeline
- Week 1: Project Setup and Data Understanding
- Week 2: Data Cleaning & Merging
- Week 3: Exploratory Data Analysis (EDA)
- Week 4: Feature Engineering & Encoding
- Week 5: Handle Class Imbalance
- Week 6: Baseline Models
- Week 7: Ensemble Model (Random Forest)
- Week 8: Model Selection & Final Evaluation
- Week 9: Interpretability & Business Insights
- Week 10: Documentation & Final Report

## Data to Collect and How
Data is collected publically and is available in kaggle (https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification/data), it will be broken into 
- `datasets/car_features.csv` (vehicle attributes)
- `datasets/insurance_claim.csv` (claim label per policy)
- `datasets/policy_features.csv` (policyholder and policy attributes)

The datasets will be merged on `policy_id`, cleaned (drop nulls and unused columns), and encoded for modeling.

## Modeling Plan 
Models planned:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Key steps include one-hot encoding categorical features, handling class imbalance with balanced sampling on train/validation/test splits, and evaluating with accuracy, ROC curves, and confusion matrices.

## Visualization Plan 
Exploratory and evaluation visuals planned in the runbook:
- Count plots for key categorical features vs. `is_claim`
- Correlation heatmaps
- Pair plots for selected numeric features
- Confusion matrix heatmaps
- ROC curves for each classifier
- Train/test accuracy bar charts to compare models

## Test Plan 
- Split the full dataset into train/validation/test (approx. 70/15/15) using random sampling.
- Address class imbalance by balancing the training set and evaluating on balanced validation/test sets for fair model comparison.
- Report accuracy, precision/recall, ROC-AUC, and confusion matrices.
