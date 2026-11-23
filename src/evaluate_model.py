"""
Final Model Training, Test Set Evaluation, and Model Serialization
Phase 6 & 7: Final Model Training, Testing, and Saving
"""
import os
import sys
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))
from train_model import load_feature_matrix, define_candidate_models

def train_and_evaluate_final_model(
    model_name: str,
    train_csv: Path,
    test_csv: Path,
    output_model_path: Path,
    results_dir: Path
):
    """
    Trains the chosen best model on the complete training set,
    evaluates on the strictly unseen test set,
    generates evaluation visualizations, and serializes the trained pipeline with metadata.
    """
    vis_dir = results_dir / "visualizations"
    eval_dir = results_dir / "evaluation"
    vis_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 6: Final Model Training and Evaluation ({model_name})")
    print("=" * 60)

    # 1. Load Train and Test feature datasets
    X_train, y_train, feature_cols, train_meta = load_feature_matrix(train_csv)
    X_test, y_test, _, test_meta = load_feature_matrix(test_csv)

    # Align columns to guarantee identical feature order
    X_test = X_test[feature_cols]

    classes = sorted(list(np.unique(y_train)))
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    print(f"Feature count: {len(feature_cols)}")

    # 2. Instantiate and train candidate pipeline on FULL training data
    candidate_models = define_candidate_models()
    if model_name not in candidate_models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(candidate_models.keys())}")

    pipeline = candidate_models[model_name]
    print(f"\nTraining '{model_name}' on complete training set...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # 3. Evaluate strictly on unseen TEST set
    print("\nEvaluating on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    # Calculate comprehensive metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred, labels=classes)
    cls_report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    cls_report_text = classification_report(y_test, y_pred, labels=classes, zero_division=0)

    print("\n" + "=" * 60)
    print(f"FINAL TEST SET EVALUATION METRICS: {model_name}")
    print("=" * 60)
    print(f"  Accuracy                       : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy              : {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  Macro Precision                : {prec_macro:.4f}")
    print(f"  Weighted Precision             : {prec_weighted:.4f}")
    print(f"  Macro Recall                   : {rec_macro:.4f}")
    print(f"  Weighted Recall                : {rec_weighted:.4f}")
    print(f"  Macro F1-Score                 : {f1_macro:.4f}")
    print(f"  Weighted F1-Score              : {f1_weighted:.4f}")
    print(f"  Matthews Correlation (MCC)    : {mcc:.4f}")

    print("\nClassification Report:\n")
    print(cls_report_text)

    # 4. Generate Visualizations
    # A. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix on Test Dataset\n{model_name} (Acc: {acc*100:.1f}%)", fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("Ground Truth Class", fontweight="bold")
    plt.tight_layout()
    cm_path = vis_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # B. Per-Class Metrics Bar Chart
    per_class_df = pd.DataFrame({
        "Class": classes,
        "Precision": [cls_report[c]["precision"] for c in classes],
        "Recall": [cls_report[c]["recall"] for c in classes],
        "F1-Score": [cls_report[c]["f1-score"] for c in classes],
        "Support": [int(cls_report[c]["support"]) for c in classes]
    })

    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(classes))
    width = 0.25
    plt.bar(x - width, per_class_df["Precision"], width, label="Precision", color="#3498db")
    plt.bar(x, per_class_df["Recall"], width, label="Recall", color="#2ecc71")
    plt.bar(x + width, per_class_df["F1-Score"], width, label="F1-Score", color="#e74c3c")
    plt.xlabel("Yoga Pose Class", fontweight="bold")
    plt.ylabel("Metric Score", fontweight="bold")
    plt.title(f"Per-Class Performance on Test Set ({model_name})", fontweight="bold", pad=15)
    plt.xticks(x, [f"{c}\n(N={s})" for c, s in zip(classes, per_class_df["Support"])], fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, facecolor="white")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for i in range(len(classes)):
        plt.text(i + width, per_class_df["F1-Score"].iloc[i] + 0.02,
                 f"{per_class_df['F1-Score'].iloc[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    pc_path = vis_dir / "per_class_metrics.png"
    plt.savefig(pc_path, dpi=300)
    plt.close()
    print(f"Saved: {pc_path}")

    # 5. Save Artifact and Comprehensive Report
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": model_name,
        "pipeline": pipeline,
        "classes": classes,
        "feature_names": feature_cols,
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": prec_macro,
            "weighted_precision": prec_weighted,
            "macro_recall": rec_macro,
            "weighted_recall": rec_weighted,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "mcc": mcc
        }
    }
    joblib.dump(bundle, output_model_path)
    print(f"\nSerialized best model bundle saved to:\n  {output_model_path}")

    report_payload = {
        "model_name": model_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": bundle["metrics"],
        "confusion_matrix": conf_mat.tolist(),
        "per_class_metrics": per_class_df.to_dict(orient="records"),
        "classification_report": cls_report
    }

    report_json_path = eval_dir / "test_evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Final test evaluation report saved to:\n  {report_json_path}")

    return bundle, report_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model on test set and save artifact")
    parser.add_argument("--model-name", type=str, default=None, help="Name of model to train and evaluate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    train_csv = project_root / "features" / "train_features.csv"
    test_csv = project_root / "features" / "test_features.csv"
    comp_csv = project_root / "results" / "model_comparison.csv"
    output_model = project_root / "models" / "best_yoga_pose_model.joblib"
    results_dir = project_root / "results"

    # If model_name is not provided, read the top performer from model_comparison.csv
    if args.model_name is None:
        if comp_csv.exists():
            comp_df = pd.read_csv(comp_csv)
            best_model_name = comp_df.iloc[0]["Model"]
            print(f"Auto-selected top cross-validation model from {comp_csv}: {best_model_name}")
        else:
            best_model_name = "Random Forest"
            print(f"Defaulting to {best_model_name}")
    else:
        best_model_name = args.model_name

    train_and_evaluate_final_model(
        model_name=best_model_name,
        train_csv=train_csv,
        test_csv=test_csv,
        output_model_path=output_model,
        results_dir=results_dir
    )

if __name__ == "__main__":
    main()
