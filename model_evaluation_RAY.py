"""
FUTURE ENHANCEMENT: 

Confidence Learning
    Methods:
    1. Calibration Plots:   
        - Evaluate whether the predicted probabilities match the actual likelihood of correctness.
        - Use tools like sklearn.calibration.calibration_curve.
    2. Uncertainty Analysis:
        - Identify predictions with low confidence (e.g., probabilities close to 0.5) and flag them for human review or further analysis.
    3. Custom Thresholding:
        - Experiment with different thresholds for classification to optimize for specific business goals (e.g., minimizing false negatives).
    Why:
    - It ensures that the model's predictions are not only accurate but also reliable.
    - It helps in making informed decisions, especially in high-stakes applications like lead scoring, where false positives or negatives can have significant business impacts.

Slicing
    Why:
    - Provides insights into model performance across different segments of the data.
    - Fairness: Ensures the model is not biased against specific subgroups.
    - Debugging: Helps identify where the model struggles and why.
    - Business Impact: Ensures consistent performance across key customer segments.

Interpretability
    Methods:
    - feature_importance
    - shap
    - pdps
    Why:
    - Trust: Helps stakeholders understand and trust the model's decisions.
    - Debugging: Identifies issues like over-reliance on irrelevant features.
    - Fairness: Ensures the model is not biased against specific groups.

Behavioral Testing
    Methods:
    - Resuse [Slicing | Confidence Learning | Interpretability]
    - Edge Case Testing
    - Bias Testing
    - Threshold Sensitivity Testing
    Why:
    - Fairness: Ensures the model performs equitably across all subgroups.
    - Robustness: Verifies that the model handles edge cases and unusual inputs gracefully.
    - Business Alignment: Confirms that the model's behavior aligns with business goals and expectations.

Capability vs Alignment
1. Capability
    - Definition: Capability refers to the technical performance of the model—how well it can achieve the task it was designed for.
    - Focus: Metrics, accuracy, and predictive power.
    - Examples:
        - Accuracy, precision, recall, F1 score, ROC AUC, etc.
        - Evaluating the model's ability to generalize to unseen data.
        - Testing the model's robustness to edge cases or noisy inputs.

2. Alignment
    - Definition: Alignment refers to how well the model's behavior matches the goals, values, and expectations of the stakeholders or the business.
    - Focus: Fairness, interpretability, and ethical considerations.
    - Examples:
        - Ensuring the model is not biased against specific subgroups (e.g., fairness across country_code or email_provider).
        - Verifying that the model's predictions align with business objectives (e.g., minimizing false negatives in lead scoring).
        - Providing explanations for predictions to build trust with stakeholders.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, 
    recall_score, accuracy_score, average_precision_score
)
import xgboost as xgb

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path, metadata_path=None):
        """Initialize evaluator with model and metadata paths"""
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.evaluation_results = {}
        self.output_dir = None  # Directory to store evaluation outputs
        
    def create_output_directory(self):
        """Create a dedicated directory for this evaluation run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_path).replace(".pkl", "")
        self.output_dir = f"results/evaluation/{model_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 Created output directory: {self.output_dir}")
        
    def load_model_and_metadata(self):
        """Load the trained model and its metadata"""
        # Load model
        self.model = joblib.load(self.model_path)
        print(f"✅ Model loaded from: {self.model_path}")
        
        # Load metadata if available
        if self.metadata_path and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Metadata loaded from: {self.metadata_path}")
            print(f"Model trained: {self.metadata.get('timestamp', 'Unknown')}")
        else:
            print("⚠️ No metadata file found")
    
    def prepare_data(self):
        """Prepare evaluation datasets"""
        print("\n" + "="*50)
        print("PREPARING EVALUATION DATA")
        print("="*50)
        
        # Load the test set from CSV files
        self.X_test = pd.read_csv("data/X_test.csv")
        self.y_test = pd.read_csv("data/y_test.csv")
        
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {self.X_test.shape[1]}")
        
    def generate_predictions(self):
        """Generate predictions on test set"""
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        # Predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("✅ Predictions generated")
        
    def evaluate_classification_metrics(self):
        """Calculate comprehensive classification metrics"""
        print("\n" + "="*50)
        print("CLASSIFICATION METRICS")
        print("="*50)
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Store results
        self.evaluation_results.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        })
        
        # Print results
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1 Score:          {f1:.4f}")
        print(f"ROC AUC:           {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Confusion matrix saved: {output_path}")
        
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ ROC curve saved: {output_path}")
        
    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "precision_recall_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Precision-Recall curve saved: {output_path}")
        
    def analyze_feature_importance(self):
        """Analyze and plot feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = self.X_test.columns if hasattr(self.X_test, 'columns') else [f'feature_{i}' for i in range(len(importance))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Top 10 features
        print("Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Feature importance saved: {output_path}")
        
        self.evaluation_results['feature_importance'] = importance_df.to_dict('records')
        
    def save_evaluation_report(self):
        """Save comprehensive evaluation report"""
        print("\n" + "="*50)
        print("SAVING EVALUATION REPORT")
        print("="*50)
        
        # Create evaluation report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'metadata_path': self.metadata_path,
            'model_metadata': self.metadata,
            'evaluation_results': self.evaluation_results,
            'test_set_size': len(self.y_test)
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Evaluation report saved: {report_path}")
        
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting Comprehensive Model Evaluation")
        print("="*60)
        
        self.create_output_directory()  # Create output directory
        self.load_model_and_metadata()
        self.prepare_data()
        self.generate_predictions()
        self.evaluate_classification_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.analyze_feature_importance()
        self.save_evaluation_report()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)

# MAIN EXECUTION
if __name__ == "__main__":
    # Find the latest model file
    model_files = [f for f in os.listdir('models/ray') if f.startswith('xgboost_ray_best_') and f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found in models/ray directory")
        exit(1)
    
    # Get the latest model
    latest_model = sorted(model_files)[-1]
    model_path = f"models/ray/{latest_model}"
    
    # Find corresponding metadata file
    timestamp = latest_model.replace('xgboost_ray_best_', '').replace('.pkl', '')
    metadata_path = f"models/ray/metadata_{timestamp}.json"

    print(f"📁 Using model: {model_path}")
    if os.path.exists(metadata_path):
        print(f"📄 Using metadata: {metadata_path}")
    else:
        print("⚠️ No metadata found for the model.")
    
    # Run evaluation
    evaluator = ModelEvaluator(model_path, metadata_path)
    evaluator.run_full_evaluation()