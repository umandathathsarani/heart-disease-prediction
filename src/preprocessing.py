import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_split_data(filepath="data/heart_disease.csv", test_size=0.2, random_state=42):
    """
    Loads the dataset, separates features and target, and splits into train and test sets.
    Ensures that the target class distribution is maintained using stratify.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split into training and testing sets
    # Using stratify=y ensures the train and test sets have the same proportion of heart disease cases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer to preprocess the data.
    - Numerical features: Impute missing values with median, then scale using StandardScaler.
    - Categorical features: Impute missing values with most frequent, then One-Hot Encode.
    
    This prevents data leakage because the pipeline will only be fitted on the training data.
    """
    # Define feature groups based on our Exploratory Data Analysis
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # 1. Pipeline for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 3. Combine both pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
