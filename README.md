# Yoga Pose Recognition Using MediaPipe Pose Estimation and Machine Learning

An end-to-end Computer Vision and Machine Learning system for recognizing and classifying human yoga poses from single RGB images using Google MediaPipe Pose Landmarker (33 body keypoints), geometric/biomechanical feature engineering, and ensemble machine learning classifiers.

---

## 📌 Project Architecture

```
                                  [ Input Yoga Image ]
                                           │
                                           ▼
                       [ MediaPipe PoseLandmarker (33 Landmarks) ]
                                           │
                                           ▼
                         [ Biomechanical Feature Engineering ]
                     ├─ Body-centered translation & scale normalization
                     ├─ 3D & 2D joint angles (elbows, knees, hips, shoulders)
                     ├─ Torso inclination to gravity vector
                     ├─ Inter-limb spread & pairwise distance ratios
                     └─ Bounding-box aspect ratio & visibility scores
                                           │ (202 Features)
                                           ▼
                       [ Scikit-Learn Preprocessing Pipeline ]
                                  (Imputer + Scaler)
                                           │
                                           ▼
                   [ Trained Classifier: Gradient Boosting / Random Forest ]
                                           │
                                           ▼
                     [ Predicted Pose Class + Confidence Score ]
                         (downdog, goddess, plank, tree, warrior2)
```

---

## 📂 Project Directory Structure

```
gesture-recognition-project/
│
├── DATASET/
│   ├── TRAIN/                  # 1,081 images across 5 classes
│   │   ├── downdog/
│   │   ├── goddess/
│   │   ├── plank/
│   │   ├── tree/
│   │   └── warrior2/
│   └── TEST/                   # 470 images across 5 classes
│       ├── downdog/
│       ├── goddess/
│       ├── plank/
│       ├── tree/
│       └── warrior2/
│
├── features/
│   ├── train_features.csv      # Extracted 202 features for 1,081 training samples
│   └── test_features.csv       # Extracted 202 features for 470 testing samples
│
├── models/
│   ├── best_yoga_pose_model.joblib   # Serialized top model pipeline + metadata
│   └── pose_landmarker_heavy.task    # MediaPipe Pose Landmarker model bundle
│
├── results/
│   ├── dataset_analysis/
│   │   ├── dataset_inspection_report.json  # Dataset distribution, formats, resolutions
│   │   └── detection_failures.csv          # Audit log of failed pose detections
│   ├── evaluation/
│   │   ├── cross_validation_detailed_results.json # Per-fold metrics across models
│   │   └── test_evaluation_report.json            # Final unseen test evaluation metrics
│   ├── visualizations/
│   │   ├── class_distribution.png          # Train/Test sample counts per class
│   │   ├── canonical_mean_skeletons.png    # Mean 2D skeletal shape per pose
│   │   ├── joint_angles_distribution.png   # Boxplots of key joint angles
│   │   ├── feature_correlation_heatmap.png # Pearson correlation of engineered features
│   │   ├── detection_failure_analysis.png  # Detection failure rate per pose
│   │   ├── sample_pose_landmarks_overlay.png # 33 keypoints drawn on sample images
│   │   ├── cv_model_comparison.png         # 5-fold CV comparison across 5 algorithms
│   │   ├── confusion_matrix.png            # Normalized confusion matrix heatmap
│   │   └── per_class_metrics.png           # Precision, Recall, F1 per class
│   └── model_comparison.csv                # Tabular cross-validation summary
│
├── src/
│   ├── data_inspection.py      # Phase 1: Dataset health and format audit
│   ├── feature_engineering.py  # Geometric, angular, and normalized feature extraction
│   ├── pose_extraction.py      # Phase 2: MediaPipe batch pose extractor
│   ├── exploratory_analysis.py # Phase 3: Exploratory data analysis & visualizations
│   ├── train_model.py          # Phase 4 & 5: Model training & 5-Fold Stratified CV
│   ├── evaluate_model.py       # Phase 6 & 7: Test set evaluation & model saving
│   └── predict.py              # Standalone inference on unseen images
│
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## ⚡ Quick Start & Reproduction

### 1. Environment Setup

Activate the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dataset Inspection (Phase 1)

Audit images, class distribution, corrupt files, and resolutions:

```powershell
python src/data_inspection.py
```

### 3. Pose Keypoint & Feature Extraction (Phase 2)

Extract 33 MediaPipe body landmarks and 202 engineered features from all images:

```powershell
python src/pose_extraction.py
```

### 4. Exploratory Data Analysis & Visualizations (Phase 3)

Generate skeletal comparison plots, correlation heatmaps, and angle distributions:

```powershell
python src/exploratory_analysis.py
```

### 5. Cross-Validation & Model Comparison (Phase 4 & 5)

Train and evaluate 5 candidate ML algorithms using 5-Fold Stratified Cross-Validation:

```powershell
python src/train_model.py
```

### 6. Final Model Evaluation & Saving (Phase 6 & 7)

Train the winning model on the full training set and evaluate on the unseen test set:

```powershell
python src/evaluate_model.py --model-name "Gradient Boosting"
```

### 7. Inference on New Images (Phase 7)

Predict yoga poses on any unseen image with confidence score and optional visual skeleton overlay:

```powershell
# Text output with probability bars
python src/predict.py --image "DATASET/TEST/tree/00000000.jpg"

# Save annotated image with skeleton overlay
python src/predict.py --image "DATASET/TEST/warrior2/00000000.jpg" --visualize "results/visualizations/pred_warrior2.jpg"

# JSON output for API integration
python src/predict.py --image "DATASET/TEST/downdog/00000000.jpg" --json
```

---

## 📊 Summary of Experimental Results

### 5-Fold Stratified Cross-Validation (Training Set: 1,035 samples)

| Model | Accuracy (Mean ± Std) | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **0.9478 ± 0.0168** | **0.9503** | **0.9451** | **0.9467** | **0.9477** |
| Random Forest | 0.9401 ± 0.0145 | 0.9458 | 0.9365 | 0.9394 | 0.9399 |
| Support Vector Machine (RBF) | 0.9324 ± 0.0097 | 0.9383 | 0.9302 | 0.9328 | 0.9324 |
| K-Nearest Neighbors | 0.9314 ± 0.0142 | 0.9358 | 0.9267 | 0.9301 | 0.9310 |
| Decision Tree | 0.9024 ± 0.0205 | 0.9016 | 0.9011 | 0.9004 | 0.9025 |

### Final Held-Out Test Set Performance (Unseen Test Set: 464 samples)

- **Overall Test Accuracy**: **99.57%** (462 / 464 correctly classified)
- **Balanced Accuracy**: **99.54%**
- **Macro Precision**: **0.9964**
- **Macro Recall**: **0.9954**
- **Macro F1-Score**: **0.9959**
- **Weighted F1-Score**: **0.9957**
- **Matthews Correlation Coefficient (MCC)**: **0.9946**

| Class | Precision | Recall | F1-Score | Support (Test Images) |
|---|:---:|:---:|:---:|:---:|
| **downdog** | 1.00 | 0.99 | 0.99 | 94 |
| **goddess** | 1.00 | 0.99 | 0.99 | 80 |
| **plank** | 0.99 | 1.00 | 1.00 | 115 |
| **tree** | 1.00 | 1.00 | 1.00 | 68 |
| **warrior2** | 0.99 | 1.00 | 1.00 | 107 |
| **Macro Average** | **1.00** | **1.00** | **1.00** | **464** |
