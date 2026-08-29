# MSc Customer Churn Prediction

Code for the MSc dissertation:

**The Impact of Feature Engineering and Model Choice on Customer Churn Prediction**

## Overview

This project investigates the impact of machine learning model choice and feature engineering methods on customer churn prediction using the Telco Customer Churn dataset.

Five machine learning models are initially compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

Logistic Regression, Random Forest and XGBoost are then selected for further feature engineering and validation experiments.

## Project Structure

- `src/` - data processing, models and experiment functions
- `scripts/` - scripts used to run the experiments
- `results/experiments/` - experimental results
- `results/figures/` - figures used in the dissertation

## Dataset

The project uses the Telco Customer Churn dataset:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place the dataset in the `data/` directory before running the scripts.

## Main Experiments

1. Data cleaning and preprocessing
2. Baseline model comparison
3. Hyperparameter experiments
4. Five-fold cross-validation
5. Feature engineering experiments
6. Permutation Importance analysis
7. Repeated data-split validation

## Requirements

See `requirements.txt`.
