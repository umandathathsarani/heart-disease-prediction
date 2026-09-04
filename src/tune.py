import os
import joblib
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.preprocessing import load_and_split_data, create_preprocessor

def tune_and_select_best_model():
    """
    Performs hyperparameter tuning on the top models (Logistic Regression and SVM),
    selects the best model based on Recall, and saves the full pipeline.
    """
    # Suppress warnings for cleaner terminal output
    warnings.filterwarnings('ignore')
    
    print("Loading ONLY training data for hyperparameter tuning to avoid data leakage...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    preprocessor = create_preprocessor()
    
    # 1. Setup Logistic Regression Pipeline and Grid
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    
    # 2. Setup SVM Pipeline and Grid
    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, probability=True))
    ])
    
    svm_param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }
    
    print("Tuning Logistic Regression...")
    # Scoring by 'recall' because missing a heart disease patient is highly dangerous
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='recall', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    
    print("Tuning Support Vector Machine...")
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='recall', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    
    print("\n==========================================")
    print("--- Tuning Results (5-Fold CV Recall) ---")
    print(f"Best Logistic Regression Recall: {lr_grid.best_score_:.4f}")
    print(f"Best SVM Recall:                 {svm_grid.best_score_:.4f}")
    
    # Select the overall best model
    if lr_grid.best_score_ >= svm_grid.best_score_:
        print("\n*** Selected Model: Logistic Regression ***")
        print(f"Optimal Parameters: {lr_grid.best_params_}")
        best_model = lr_grid.best_estimator_
    else:
        print("\n*** Selected Model: Support Vector Machine ***")
        print(f"Optimal Parameters: {svm_grid.best_params_}")
        best_model = svm_grid.best_estimator_
        
    print("\n--- Medical Classification Trade-offs ---")
    print("Why did we prioritize Recall over Accuracy during tuning?")
    print("- Recall: Minimizes False Negatives. In medicine, telling a sick patient they are healthy is extremely dangerous.")
    print("- Precision: Minimizes False Positives. Telling a healthy patient they are sick causes stress, but usually just leads to further testing.")
    print("Therefore, our hyperparameter search strictly optimized for Recall.")
    
    # Save the full pipeline
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model pipeline successfully saved to {model_path}")

if __name__ == "__main__":
    tune_and_select_best_model()
