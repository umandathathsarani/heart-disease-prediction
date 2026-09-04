import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# Import the training function so we can get the trained models and test data
from src.train import train_models

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir="images"):
    """Generates and saves a confusion matrix plot for a given model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    
    # Use seaborn heatmap for a professional look
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label (0=Healthy, 1=Disease)')
    plt.ylabel('True Label')
    
    # Create safe filename and save
    safe_name = model_name.replace(" ", "_").lower()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{safe_name}_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

def evaluate_all_models():
    """
    Evaluates all trained models, generates plots, and saves comparison metrics.
    """
    print("Starting Model Evaluation...")
    # 1. Get the trained models and the test dataset
    trained_pipelines, X_test, y_test = train_models()
    
    metrics_list = []
    
    # Setup plot for ROC Curves comparison
    plt.figure(figsize=(10, 8))
    
    for model_name, pipeline in trained_pipelines.items():
        print(f"\n==========================================")
        print(f"Evaluating {model_name}...")
        print(f"==========================================")
        
        # 2. Make Predictions
        y_pred = pipeline.predict(X_test)
        
        # For ROC-AUC, we need probabilities of the positive class (1)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)
            
        # 3. Calculate Core Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics_list.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })
        
        # 4. Generate Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, model_name)
        
        # 5. Print Classification Report (Terminal Output)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # 6. Add to ROC Curve plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
    # Finalize and Save ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/roc_curves_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Save the model comparison table
    os.makedirs('reports', exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort primarily by Recall (critical for medical data) and F1 Score
    metrics_df = metrics_df.sort_values(by=['Recall', 'F1 Score'], ascending=False)
    metrics_df.to_csv('reports/model_comparison.csv', index=False)
    
    print("\nEvaluation Complete!")
    print("\n--- Model Comparison Table ---")
    print(metrics_df.to_string(index=False))
    print("\nAll plots saved to 'images/' and comparison table saved to 'reports/model_comparison.csv'")

if __name__ == "__main__":
    evaluate_all_models()
