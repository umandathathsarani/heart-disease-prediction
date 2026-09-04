import argparse
from src.utils import download_and_prepare_dataset
from src.train import train_models
from src.evaluate import evaluate_all_models
from src.tune import tune_and_select_best_model
from src.predict import make_prediction

def main():
    print("==================================================")
    print("       HEART DISEASE PREDICTION PIPELINE")
    print("==================================================")
    print("This is an educational project. Not for medical diagnosis.\n")
    
    parser = argparse.ArgumentParser(description="Run the Heart Disease Prediction ML Pipeline.")
    parser.add_argument(
        '--step', 
        type=str, 
        choices=['all', 'download', 'train', 'evaluate', 'tune', 'predict'],
        default='all',
        help="Which step of the ML workflow to run. Default is 'all'."
    )
    
    args = parser.parse_args()
    
    if args.step in ['all', 'download']:
        print("\n---> [1/5] Downloading and Preparing Dataset...")
        download_and_prepare_dataset()
        
    if args.step in ['all', 'train']:
        print("\n---> [2/5] Training Baseline Models...")
        train_models()
        
    if args.step in ['all', 'evaluate']:
        print("\n---> [3/5] Evaluating Models & Generating Reports...")
        evaluate_all_models()
        
    if args.step in ['all', 'tune']:
        print("\n---> [4/5] Tuning Hyperparameters & Saving Best Model...")
        tune_and_select_best_model()
        
    if args.step in ['all', 'predict']:
        print("\n---> [5/5] Testing Prediction Script on Sample Patient...")
        sample_patient = {
            "age": 60, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, 
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, 
            "slope": 0, "ca": 0, "thal": 1
        }
        make_prediction(sample_patient)
        
    print("\n==================================================")
    print("Workflow Execution Complete!")
    print("==================================================")

if __name__ == "__main__":
    main()
