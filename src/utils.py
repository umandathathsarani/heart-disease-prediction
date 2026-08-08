import pandas as pd
import os

def download_and_prepare_dataset(output_path="data/heart_disease.csv"):
    """
    Downloads the UCI Heart Disease dataset (Cleveland), cleans it,
    and saves it to the specified output path.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Standard columns based on UCI repository description
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    print(f"Downloading dataset from {url}...")
    # '?' represents missing values in this specific dataset
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # In the original dataset, target ranges from 0 (no disease) to 4. 
    # For a binary classification, we map > 0 to 1 (presence of disease).
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    download_and_prepare_dataset()
