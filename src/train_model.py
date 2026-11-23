"""
Machine Learning Model Training and Stratified K-Fold Cross-Validation
Phase 4 & 5: Model Development, Cross-Validation, and Comparison
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

def load_feature_matrix(csv_path: Path):
    """
    Loads feature dataset and extracts numeric feature matrix X, target labels y,
    and metadata (image names, paths).
    Filters out detection failures so only valid poses are modeled.
    """
    df = pd.read_csv(csv_path)
    
    # Filter only successful detections
    valid_df = df[df["detection_success"] == True].copy()
    
    metadata_cols = ["image_name", "image_path", "split", "label", "detection_success", "detection_error"]
    raw_cols = [c for c in df.columns if "_raw_" in c]
    exclude_cols = set(metadata_cols + raw_cols)
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = valid_df[feature_cols].copy()
    y = valid_df["label"].values
    
    return X, y, feature_cols, valid_df


def define_candidate_models():
    """
    Returns candidate models configured with Scikit-learn Pipelines.
    All feature scaling and imputation are strictly confined inside the pipeline.
    """
    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.85,
                random_state=RANDOM_STATE
            ))
        ])
    }
    return models


def evaluate_with_cross_validation(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
    """
    Executes Stratified K-Fold Cross-Validation for each candidate model.
    Computes per-fold Accuracy, Precision, Recall, and F1-score (Macro and Weighted).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = define_candidate_models()
    
    cv_results = {}
    summary_records = []
    
    print(f"\nRunning {n_splits}-Fold Stratified Cross-Validation on {len(y)} samples...")
    print(f"Target Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  - {cls:<12}: {cnt} samples ({cnt/len(y)*100:.1f}%)")
        
    for name, pipeline in models.items():
        print(f"\nEvaluating: {name}...")
        fold_metrics = {
            "accuracy": [],
            "precision_macro": [],
            "precision_weighted": [],
            "recall_macro": [],
            "recall_weighted": [],
            "f1_macro": [],
            "f1_weighted": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            prec_m = precision_score(y_val, y_pred, average="macro", zero_division=0)
            prec_w = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            rec_m = recall_score(y_val, y_pred, average="macro", zero_division=0)
            rec_w = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            f1_m = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            
            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision_macro"].append(prec_m)
            fold_metrics["precision_weighted"].append(prec_w)
            fold_metrics["recall_macro"].append(rec_m)
            fold_metrics["recall_weighted"].append(rec_w)
            fold_metrics["f1_macro"].append(f1_m)
            fold_metrics["f1_weighted"].append(f1_w)
            
            print(f"  Fold {fold}: Acc={acc:.4f}, Macro-F1={f1_m:.4f}, Weighted-F1={f1_w:.4f}")
            
        cv_results[name] = fold_metrics
        
        summary_records.append({
            "Model": name,
            "Accuracy_Mean": np.mean(fold_metrics["accuracy"]),
            "Accuracy_Std": np.std(fold_metrics["accuracy"]),
            "Macro_Precision_Mean": np.mean(fold_metrics["precision_macro"]),
            "Macro_Precision_Std": np.std(fold_metrics["precision_macro"]),
            "Macro_Recall_Mean": np.mean(fold_metrics["recall_macro"]),
            "Macro_Recall_Std": np.std(fold_metrics["recall_macro"]),
            "Macro_F1_Mean": np.mean(fold_metrics["f1_macro"]),
            "Macro_F1_Std": np.std(fold_metrics["f1_macro"]),
            "Weighted_F1_Mean": np.mean(fold_metrics["f1_weighted"]),
            "Weighted_F1_Std": np.std(fold_metrics["f1_weighted"])
        })
        
    summary_df = pd.DataFrame(summary_records).sort_values(by="Macro_F1_Mean", ascending=False)
    return cv_results, summary_df, models


def plot_cv_comparison(summary_df: pd.DataFrame, out_file: Path):
    """Generates comparison bar chart with standard deviation error bars."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(summary_df))
    width = 0.22
    
    plt.bar(x - width, summary_df["Accuracy_Mean"], width, yerr=summary_df["Accuracy_Std"],
            label="Accuracy", capsize=4, color="#3498db", alpha=0.9)
    plt.bar(x, summary_df["Macro_F1_Mean"], width, yerr=summary_df["Macro_F1_Std"],
            label="Macro F1-Score", capsize=4, color="#2ecc71", alpha=0.9)
    plt.bar(x + width, summary_df["Weighted_F1_Mean"], width, yerr=summary_df["Weighted_F1_Std"],
            label="Weighted F1-Score", capsize=4, color="#9b59b6", alpha=0.9)
    
    plt.xlabel("Machine Learning Algorithm", fontweight="bold", fontsize=12)
    plt.ylabel("Score (0.0 - 1.0)", fontweight="bold", fontsize=12)
    plt.title("5-Fold Stratified Cross-Validation Model Comparison", fontweight="bold", fontsize=14, pad=15)
    plt.xticks(x, summary_df["Model"], rotation=15, ha="right", fontsize=11)
    plt.ylim(0.7, 1.02)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Annotate values
    for i in range(len(summary_df)):
        plt.text(i, summary_df["Macro_F1_Mean"].iloc[i] + 0.015,
                 f"{summary_df['Macro_F1_Mean'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
                 
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    train_csv = project_root / "features" / "train_features.csv"
    results_dir = project_root / "results"
    vis_dir = results_dir / "visualizations"
    eval_dir = results_dir / "evaluation"
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_csv.exists():
        print(f"Error: {train_csv} does not exist. Run pose_extraction.py first.")
        sys.exit(1)
        
    print("=" * 60)
    print("PHASE 4 & 5: Model Development & Stratified Cross-Validation")
    print("=" * 60)
    
    X, y, feature_cols, valid_df = load_feature_matrix(train_csv)
    print(f"Loaded {len(X)} valid training samples with {len(feature_cols)} engineered features.")
    
    cv_results, summary_df, models = evaluate_with_cross_validation(X, y, n_splits=5)
    
    # Save comparison table
    comp_csv = results_dir / "model_comparison.csv"
    summary_df.to_csv(comp_csv, index=False)
    print(f"\nModel Comparison Table saved to: {comp_csv}")
    
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS SUMMARY (Sorted by Macro F1-Score)")
    print("=" * 60)
    print(summary_df[["Model", "Accuracy_Mean", "Macro_Precision_Mean", "Macro_Recall_Mean", "Macro_F1_Mean", "Weighted_F1_Mean"]].to_string(index=False))
    
    best_model_name = summary_df.iloc[0]["Model"]
    best_f1 = summary_df.iloc[0]["Macro_F1_Mean"]
    print(f"\n>>> Best Model: {best_model_name} (Macro F1 = {best_f1:.4f})")
    
    # Save detailed CV fold results
    cv_json = eval_dir / "cross_validation_detailed_results.json"
    with open(cv_json, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"Detailed CV results saved to: {cv_json}")
    
    # Plot comparison
    plot_cv_comparison(summary_df, vis_dir / "cv_model_comparison.png")

if __name__ == "__main__":
    main()
