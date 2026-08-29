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

- `src/` - data processing, model definitions and experiment functions
- `scripts/` - scripts used to run the experiments and generate figures
- `results/experiments/` - experimental results
- `results/feature_engineering/` - feature ranking results
- `results/figures/` - figures used in the dissertation

## Dataset

The project uses the Telco Customer Churn dataset:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Create a `data/` directory in the project root and place the dataset file there before running the scripts.

## Main Experiments

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Baseline model comparison
4. Hyperparameter experiments
5. Five-fold cross-validation
6. Feature engineering experiments
7. Permutation Importance analysis
8. Repeated data-split validation

## Requirements

The project was developed using Python 3.13.

Install the required Python packages using:

```bash
pip install -r requirements.txt
