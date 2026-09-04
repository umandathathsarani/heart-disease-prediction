# Heart Disease Prediction

## Overview
This project builds a professional, supervised machine learning classification system to predict whether a patient is likely to have heart disease based on clinical and demographic features.

## Disclaimer
**This is an educational machine learning project and not a medical diagnostic system.** The predictions made by these models should not be used for medical decisions or replace a qualified healthcare professional.

## Problem Statement
The goal is to classify patients into two categories (presence or absence of heart disease) using classical machine learning algorithms, demonstrating a complete ML workflow from data exploration to model evaluation.

## Dataset
- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Number of samples:** 303 patient records
- **Features:** 13 clinical attributes
- **Target variable:** Presence (1) or absence (0) of heart disease

## Features
| Feature | Description |
|---|---|
| **age** | Age in years |
| **sex** | Sex (1 = male; 0 = female) |
| **cp** | Chest pain type (0-3) |
| **trestbps** | Resting blood pressure (in mm Hg on admission to the hospital) |
| **chol** | Serum cholesterol in mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| **restecg** | Resting electrocardiographic results (0-2) |
| **thalach** | Maximum heart rate achieved |
| **exang** | Exercise induced angina (1 = yes; 0 = no) |
| **oldpeak** | ST depression induced by exercise relative to rest |
| **slope** | The slope of the peak exercise ST segment (0-2) |
| **ca** | Number of major vessels (0-3) colored by fluoroscopy |
| **thal** | 1 = normal; 2 = fixed defect; 3 = reversible defect |

## Machine Learning Workflow
Dataset → Data Understanding → Exploratory Data Analysis (EDA) → Data Cleaning → Preprocessing → Train/Test Split → Model Training → Model Comparison → Evaluation → Hyperparameter Tuning → Model Selection → Prediction

## Models
We trained and compared the following algorithms:
- **Logistic Regression:** A linear model predicting probabilities using a logistic function. Highly interpretable.
- **Decision Tree:** A non-linear model splitting data based on feature thresholds.
- **Random Forest:** An ensemble of decision trees to reduce overfitting and improve accuracy.
- **K-Nearest Neighbors (KNN):** Classifies a patient based on the majority class of their 'k' closest neighbors.
- **Support Vector Machine (SVM):** Finds the optimal hyperplane that separates the classes in high-dimensional space.

## Evaluation Metrics
- **Recall (Primary Metric):** Proportion of actual positives identified correctly. *Crucial in medicine to minimize False Negatives.*
- **Accuracy:** Overall correctness of the model.
- **Precision:** Proportion of positive identifications that were actually correct.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under the receiver operating characteristic curve.

## Results
Our hyperparameter tuning optimized strictly for **Recall**. Both Logistic Regression and SVM achieved top results. The final selected model was **Logistic Regression**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 88.52% | 83.87% | **92.86%** | 88.14% | 96.65% |
| **Support Vector Machine** | 88.52% | 83.87% | **92.86%** | 88.14% | 96.43% |
| **Random Forest** | 86.89% | 81.25% | **92.86%** | 86.67% | 94.26% |
| **K-Nearest Neighbors** | 88.52% | 86.21% | 89.29% | 87.72% | 95.29% |
| **Decision Tree** | 73.77% | 67.65% | 82.14% | 74.19% | 74.40% |

## Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate
# OR on macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
The easiest way to run the project is using the unified `main.py` hub:

```bash
# Run the entire pipeline sequentially
python main.py --step all

# Or run specific stages:
python main.py --step download    # Downloads UCI dataset
python main.py --step train       # Trains baseline models
python main.py --step evaluate    # Generates plots and reports
python main.py --step tune        # Tunes and saves best model
python main.py --step predict     # Tests a patient prediction
```
*Note: To view the Exploratory Data Analysis, open `notebooks/01_exploratory_data_analysis.ipynb` in VS Code or Jupyter and run the cells.*

## Project Structure
```text
heart-disease-prediction/
│
├── data/                  # Ignored by git, holds the CSV dataset
├── images/                # EDA and evaluation plots
├── models/                # Holds the saved best_model.pkl pipeline
├── notebooks/             # Jupyter notebook for EDA
├── reports/               # Output CSV metrics
├── src/                   # Core Python modules
│   ├── __init__.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── tune.py
│   └── utils.py
│
├── .gitignore
├── LICENSE
├── main.py                # Command-line entry point
├── README.md
└── requirements.txt
```

## Future Improvements
- Better feature engineering
- More extensive cross-validation techniques
- Explainable AI (using SHAP values)
- Deploying a web interface using FastAPI

## License
MIT License. Dataset provided by the UCI Machine Learning Repository.
