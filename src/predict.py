import pandas as pd
import joblib

def make_prediction(patient_data, model_path='models/best_model.pkl'):
    """
    Loads the trained model pipeline and predicts whether the patient has heart disease.
    
    Args:
        patient_data (dict): A dictionary containing the 13 clinical features.
        model_path (str): Path to the saved Joblib model.
    """
    try:
        # Load the fully trained pipeline (handles both preprocessing and prediction)
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{model_path}'. Please run src/tune.py first.")
        return
        
    # Convert the single patient dictionary into a pandas DataFrame
    df = pd.DataFrame([patient_data])
    
    # Generate prediction and probability
    prediction = model.predict(df)[0]
    
    if hasattr(model, "predict_proba"):
        # Get the probability of class 1 (Heart Disease)
        probability = model.predict_proba(df)[0][1] * 100
    else:
        probability = "N/A"
        
    # Display results cleanly
    print("\n==================================================")
    print("              PREDICTION RESULTS")
    print("==================================================")
    
    if prediction == 1:
        print("Predicted Class: High Risk of Heart Disease (1)")
    else:
        print("Predicted Class: Low Risk of Heart Disease (0)")
        
    if isinstance(probability, float):
        print(f"Confidence:      {probability:.1f}% probability of heart disease.")
        
    print("\n*** DISCLAIMER ***")
    print("This prediction is for educational purposes only and is not a medical diagnosis.")
    print("Always consult a qualified healthcare professional for medical advice.")
    print("==================================================\n")

if __name__ == "__main__":
    # Example hypothetical patient data
    sample_patient = {
        "age": 60,           # Age in years
        "sex": 1,            # 1 = Male, 0 = Female
        "cp": 3,             # Chest pain type (0-3)
        "trestbps": 145,     # Resting blood pressure
        "chol": 233,         # Serum cholesterol in mg/dl
        "fbs": 1,            # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        "restecg": 0,        # Resting electrocardiographic results (0-2)
        "thalach": 150,      # Maximum heart rate achieved
        "exang": 0,          # Exercise induced angina (1 = yes; 0 = no)
        "oldpeak": 2.3,      # ST depression induced by exercise relative to rest
        "slope": 0,          # The slope of the peak exercise ST segment (0-2)
        "ca": 0,             # Number of major vessels (0-3)
        "thal": 1            # 1 = normal; 2 = fixed defect; 3 = reversable defect
    }
    
    print("Analyzing sample patient data...")
    make_prediction(sample_patient)
