import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.preprocessing import load_and_split_data, create_preprocessor

def build_models():
    """
    Returns a dictionary of initialized machine learning models.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        # probability=True is needed so we can calculate ROC-AUC later
        "Support Vector Machine": SVC(random_state=42, probability=True) 
    }
    return models

def train_models(filepath="data/heart_disease.csv"):
    """
    Loads the data, sets up the preprocessing pipeline, and trains all baseline models.
    Returns a dictionary of the trained Scikit-Learn pipelines and the test sets.
    """
    # 1. Load the split data
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data(filepath=filepath)
    
    # 2. Get the preprocessor
    preprocessor = create_preprocessor()
    
    # 3. Get the baseline models
    models = build_models()
    trained_pipelines = {}
    
    # 4. Train each model within a full pipeline
    print("Training models...")
    for model_name, model in models.items():
        # Create a pipeline combining the common preprocessing steps and the specific model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit the entire pipeline on the training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        print(f" - {model_name} trained successfully.")
        
    return trained_pipelines, X_test, y_test

if __name__ == "__main__":
    trained_pipelines, X_test, y_test = train_models()
    print("\nAll baseline models trained successfully. They are ready for evaluation!")
