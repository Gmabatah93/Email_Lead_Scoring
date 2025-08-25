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


"""
Why Use a Class for Model Evaluation?
1. Stateful Operations:
    - Evaluation involves loading a model, metadata, test data, generating predictions, storing results, and saving plots/reports.
    - A class allows you to keep all this state (self.model, self.X_test, self.evaluation_results, etc.) together, making the code cleaner and less error-prone.
2. Grouping Related Methods:
    - All evaluation steps (metrics, plots, reports) are logically related and operate on the same data.
    - A class groups these methods, so you don’t have to pass the same arguments around repeatedly.
3. Extensibility:
    - If you want to add more evaluation features (e.g., calibration, slicing, interpretability), you can easily add new methods to the class.
4. Reusability:
    - You can instantiate ModelEvaluator for different models or datasets without rewriting code.

Why Not Use Classes for Preprocessing/Training?
- Preprocessing and training in your scripts are mostly stateless, linear pipelines.
- Functions are sufficient and simpler for these steps, as you don’t need to maintain or share much state between them.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from typing import Optional, Dict, Any
from typing import Optional, Dict, Any
import typer
from typing_extensions import Annotated
import logging
import warnings

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, 
    recall_score, accuracy_score, average_precision_score
)
import xgboost as xgb

app = typer.Typer()

# Suppress most logging and warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class ModelEvaluator:
    """
    Comprehensive model evaluation class for XGBoost models.
    Handles loading models, preparing data, generating predictions,
    computing metrics, plotting, and saving evaluation reports.
    """

    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the evaluator with model and metadata paths.

        Args:
            model_path (str): Path to the trained model file (.pkl).
            metadata_path (Optional[str]): Path to the model metadata JSON file.
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.evaluation_results = {}
        self.output_dir = None  # Directory to store evaluation outputs
        
    def create_output_directory(self) -> None:
        """
        Create a dedicated directory for this evaluation run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_path).replace(".pkl", "")
        self.output_dir = f"results/evaluation/{model_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        typer.echo(typer.style(f"📂 Created output directory: {self.output_dir}", fg=typer.colors.BRIGHT_GREEN))

    def load_model_and_metadata(self) -> None:
        """
        Load the trained model and its metadata (if available).
        """
        self.model = joblib.load(self.model_path)
        typer.echo(typer.style(f"✅ Model loaded from: {self.model_path}", fg=typer.colors.BRIGHT_GREEN))

        # Load metadata if available
        if self.metadata_path and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            typer.echo(typer.style(f"✅ Metadata loaded from: {self.metadata_path}", fg=typer.colors.BRIGHT_GREEN))
        else:
            print("⚠️ No metadata file found")
    
    def prepare_data(self) -> None:
        """
        Prepare evaluation datasets by loading test features and labels.
        """
        typer.echo(typer.style("⚙️ Preparing Evaluation Data...", fg=typer.colors.BRIGHT_YELLOW))

        # Load the test set from CSV files
        self.X_test = pd.read_csv("data/X_test.csv")
        self.y_test = pd.read_csv("data/y_test.csv")

        typer.echo(f"📝 Test samples: {len(self.X_test)}")
        typer.echo(f"📝 Features: {self.X_test.shape[1]}\n")

    def generate_predictions(self) -> None:
        """
        Generate predictions and predicted probabilities on the test set.
        """
        typer.echo(typer.style("🔮 Generating Predictions...", fg=typer.colors.BRIGHT_YELLOW))

        # Predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        typer.echo(typer.style("✅ Predictions generated\n", fg=typer.colors.BRIGHT_GREEN))

    def evaluate_classification_metrics(self) -> None:
        """
        Calculate and print comprehensive classification metrics.
        Stores results in self.evaluation_results.
        """
        typer.echo(typer.style(f"📊 Classification Metrics " + "---" * 5, fg=typer.colors.BRIGHT_MAGENTA))
 
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
        typer.echo(typer.style("-" * 40, fg=typer.colors.BRIGHT_MAGENTA))

    def plot_confusion_matrix(self) -> None:
        """
        Plot and save the confusion matrix as a PNG file.
        """
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
        typer.echo(typer.style(f"✅ Confusion matrix saved: {output_path}", fg=typer.colors.BRIGHT_GREEN))

    def plot_roc_curve(self) -> None:
        """
        Plot and save the ROC curve as a PNG file.
        """
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
        typer.echo(typer.style(f"✅ ROC curve saved: {output_path}", fg=typer.colors.BRIGHT_GREEN))

    def plot_precision_recall_curve(self) -> None:
        """
        Plot and save the Precision-Recall curve as a PNG file.
        """
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
        typer.echo(typer.style(f"✅ Precision-Recall curve saved: {output_path}\n", fg=typer.colors.BRIGHT_GREEN))

    def analyze_feature_importance(self) -> None:
        """
        Analyze and plot feature importances. Saves plot and stores importances in evaluation_results.
        """
        typer.echo(typer.style(f"🔍 Feature Importance Analysis " + "---" * 4, fg=typer.colors.BRIGHT_MAGENTA))

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
        typer.echo(typer.style(f"✅ Feature importance saved: {output_path}", fg=typer.colors.BRIGHT_GREEN))

        self.evaluation_results['feature_importance'] = importance_df.to_dict('records')
        
    def save_evaluation_report(self) -> None:
        """
        Save a comprehensive evaluation report as a JSON file.
        """
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

        typer.echo(typer.style(f"✅ Evaluation report saved: {report_path}", fg=typer.colors.BRIGHT_GREEN))

    def run_full_evaluation(self) -> None:
        """
        Run the complete evaluation pipeline: directory creation, model loading,
        data preparation, prediction, metrics, plots, feature importance, and report saving.
        """
        print("="*60)
        typer.echo(typer.style("🚀 Starting Comprehensive Model Evaluation", fg=typer.colors.CYAN))
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

        typer.echo(typer.style("✅ EVALUATION COMPLETE!", fg=typer.colors.BRIGHT_GREEN))

# Typer Command
@app.command()
def main(
    model_path: Annotated[Optional[str], typer.Option(help="Path to the model file (.pkl). Defaults to the latest model.")] = None,
    metadata_path: Annotated[Optional[str], typer.Option(help="Path to the model metadata file (.json).")] = None
):
    """
    Run the comprehensive model evaluation pipeline.
    """
    if not model_path:
        model_files = [f for f in os.listdir('models/') if f.startswith('xgboost_ray_best_') and f.endswith('.pkl')]
        if not model_files:
            typer.echo("❌ No model files found in models/ray directory")
            raise typer.Exit(code=1)
        latest_model = sorted(model_files)[-1]
        model_path = f"models/{latest_model}"
        
        if not metadata_path:
            timestamp = latest_model.replace('xgboost_ray_best_', '').replace('.pkl', '')
            metadata_path = f"models/json/xgboost_ray_best_metadata_{timestamp}.json"
            if not os.path.exists(metadata_path):
                metadata_path = None

    typer.echo(typer.style(f"📁 Using model: {model_path}", fg=typer.colors.BRIGHT_RED))
    if metadata_path:
        typer.echo(typer.style(f"📄 Using metadata: {metadata_path}\n", fg=typer.colors.BRIGHT_RED))
    else:
        typer.echo("⚠️ No metadata found for the model.")
        
    evaluator = ModelEvaluator(model_path, metadata_path)
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    app()