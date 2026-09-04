<div align="center">
  
# 🫀 Heart Disease Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**A professional, supervised machine learning pipeline for classifying heart disease risk.**

</div>

---

## ⚠️ Disclaimer
> **This is an educational machine learning project and not a medical diagnostic system.** The predictions made by these models should not be used for medical decisions or replace a qualified healthcare professional.

## 📖 Overview
The goal of this project is to classify patients into two categories (presence or absence of heart disease) using classical machine learning algorithms. It demonstrates a complete end-to-end ML workflow, from data exploration and preprocessing to hyperparameter tuning and model evaluation.

## 📊 Dataset
- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Samples:** `303` patient records
- **Features:** `13` clinical attributes
- **Target:** `1` (Presence of heart disease) / `0` (Absence of heart disease)

<details>
<summary><strong>👉 Click here to expand the Feature Dictionary</strong></summary>

| Feature | Description |
|:---|:---|
| **`age`** | Age in years |
| **`sex`** | Sex (1 = male; 0 = female) |
| **`cp`** | Chest pain type (0-3) |
| **`trestbps`** | Resting blood pressure (in mm Hg on admission) |
| **`chol`** | Serum cholesterol in mg/dl |
| **`fbs`** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| **`restecg`** | Resting electrocardiographic results (0-2) |
| **`thalach`** | Maximum heart rate achieved |
| **`exang`** | Exercise induced angina (1 = yes; 0 = no) |
| **`oldpeak`** | ST depression induced by exercise relative to rest |
| **`slope`** | The slope of the peak exercise ST segment (0-2) |
| **`ca`** | Number of major vessels (0-3) colored by fluoroscopy |
| **`thal`** | 1 = normal; 2 = fixed defect; 3 = reversible defect |

</details>

## ⚙️ Machine Learning Workflow

```mermaid
graph LR
    A[Dataset] --> B[EDA]
    B --> C[Preprocessing]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Hyperparameter Tuning]
    F --> G[Prediction]
```

### 🧠 Models Evaluated
We trained and compared the following classical ML algorithms:
1. **Logistic Regression:** Highly interpretable linear model predicting probabilities.
2. **Support Vector Machine (SVM):** Finds the optimal hyperplane separating classes.
3. **Random Forest:** An ensemble of decision trees to reduce overfitting.
4. **K-Nearest Neighbors (KNN):** Classifies based on 'k' closest neighbors.
5. **Decision Tree:** A non-linear model splitting data on feature thresholds.

## 📈 Results & Evaluation
In a medical context, **Recall** is the most critical metric because it minimizes *False Negatives* (telling a sick patient they are healthy is extremely dangerous). Therefore, our hyperparameter tuning optimized strictly for Recall. 

Both Logistic Regression and SVM achieved top results. The final selected model was **Logistic Regression**.

| Model | Accuracy | Precision | 🎯 Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression** | `88.52%` | `83.87%` | **`92.86%`** | `88.14%` | `96.65%` |
| **Support Vector Machine** | `88.52%` | `83.87%` | **`92.86%`** | `88.14%` | `96.43%` |
| **Random Forest** | `86.89%` | `81.25%` | **`92.86%`** | `86.67%` | `94.26%` |
| **K-Nearest Neighbors** | `88.52%` | `86.21%` | `89.29%` | `87.72%` | `95.29%` |
| **Decision Tree** | `73.77%` | `67.65%` | `82.14%` | `74.19% | `74.40%` |

## 💻 Installation

```bash
# 1. Clone the repository
git clone https://github.com/umandathathsarani/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

The easiest way to run the project is using the unified `main.py` CLI hub:

```bash
# Run the entire pipeline sequentially from start to finish
python main.py --step all
```

Or you can run specific stages individually:
```bash
python main.py --step download    # Downloads the UCI dataset
python main.py --step train       # Trains the baseline models
python main.py --step evaluate    # Generates plots and reports
python main.py --step tune        # Tunes and saves the best model
python main.py --step predict     # Tests a sample patient prediction
```

*💡 **Tip:** To view the Exploratory Data Analysis, open `notebooks/01_exploratory_data_analysis.ipynb` in VS Code or Jupyter and run the cells!*

## 📁 Project Structure
```text
heart-disease-prediction/
├── data/                  # Ignored by git, holds the CSV dataset
├── images/                # EDA and evaluation plots
├── models/                # Holds the saved best_model.pkl pipeline
├── notebooks/             # Jupyter notebook for EDA
├── reports/               # Output CSV metrics
├── src/                   # Core Python modules
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── tune.py
│   └── utils.py
├── main.py                # Command-line entry point
└── requirements.txt
```

## 🔮 Future Improvements
- [ ] Implement robust feature engineering
- [ ] Add Explainable AI (SHAP values) for medical transparency
- [ ] Deploy a REST API backend using FastAPI
- [ ] Build a frontend web interface

## 📜 License
This project is licensed under the MIT License. The dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
