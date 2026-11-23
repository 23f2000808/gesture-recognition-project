# Yoga Pose Recognition Using MediaPipe Pose Estimation and Machine Learning: Technical Research Report

**Author:** Senior Machine Learning Engineer & Computer Vision Researcher  
**Date:** November 23, 2026  
**Project:** End-to-End Yoga Pose Classification Pipeline  

---

## Executive Summary

This report documents the design, implementation, and empirical evaluation of an end-to-end Computer Vision and Machine Learning system for recognizing human yoga poses from single RGB images. Using **Google MediaPipe PoseLandmarker (Heavy model bundle)**, 33 anatomical body keypoints with 3D coordinates and visibility/presence scores were extracted across 1,551 images. A domain-informed **Biomechanical Feature Engineering Pipeline** translated raw keypoint coordinates into 202 body-centered, translation-invariant, scale-invariant, and rotationally discriminative features (including 3D/2D joint angles, torso inclination relative to gravity, limb spread ratios, and aspect ratios).

Through **5-Fold Stratified Cross-Validation** on the training set (1,035 valid samples), five machine learning algorithms were rigorously evaluated. **Gradient Boosting** achieved the top cross-validation performance with a Macro F1-score of **0.9467 ± 0.0168**, followed by **Random Forest** (**0.9394 ± 0.0145**). Evaluated on the held-out unseen test set (464 valid samples), the final trained Gradient Boosting pipeline achieved an **Accuracy of 99.57%**, **Balanced Accuracy of 99.54%**, **Macro F1-Score of 0.9959**, and a **Matthews Correlation Coefficient (MCC) of 0.9946**, misclassifying only 2 out of 464 test instances.

---

## 1. Project Overview & Objectives

Yoga pose classification is a challenging computer vision task characterized by high intra-class variance (variations in practitioner morphology, camera angles, perspective distortion, and flexibility) and low inter-class variance between distinct postures. The primary goals of this project were:

1. **Robust Pose Detection**: Implement MediaPipe PoseLandmarker to reliably locate all 33 full-body landmarks across diverse image formats, aspect ratios, and resolutions without silent data loss.
2. **Physiologically Grounded Feature Engineering**: Construct normalized geometric descriptors and biomechanical joint angles that isolate pose configuration from subject height, body mass, camera distance, and image resolution.
3. **Leakage-Free Machine Learning**: Conduct rigorous model selection using Stratified K-Fold Cross-Validation where feature scaling and imputation are strictly confined inside training folds using Scikit-Learn pipelines.
4. **Independent Final Testing**: Evaluate the best model on an untouched, held-out test split, generating complete classification reports, confusion matrices, and per-class diagnostics.
5. **Production-Ready Inference**: Deliver a standalone, modular prediction pipeline capable of receiving raw unseen images, estimating landmarks, computing features, and returning confidence probabilities and annotated overlays.

---

## 2. Dataset Inspection & Quality Audit (Phase 1)

The dataset was located in `DATASET/` structured as directory-level class splits:

```
DATASET/
├── TRAIN/  (downdog, goddess, plank, tree, warrior2)
└── TEST/   (downdog, goddess, plank, tree, warrior2)
```

A comprehensive dataset integrity audit was conducted using OpenCV, PIL, and cryptographic hashing (MD5):

| Parameter | Training Set | Testing Set | Total |
|---|:---:|:---:|:---:|
| **Total Images** | 1,081 | 470 | **1,551** |
| **downdog** | 223 (20.6%) | 97 (20.6%) | 320 (20.6%) |
| **goddess** | 180 (16.7%) | 80 (17.0%) | 260 (16.8%) |
| **plank** | 266 (24.6%) | 115 (24.5%) | 381 (24.6%) |
| **tree** | 160 (14.8%) | 69 (14.7%) | 229 (14.8%) |
| **warrior2** | 252 (23.3%) | 109 (23.2%) | 361 (23.3%) |
| **Corrupted Files** | 0 | 0 | 0 |
| **Unreadable Files** | 0 | 0 | 0 |
| **Image Formats** | JPEG (.jpg: 1,429; .jpeg: 10), PNG (.png: 111), BMP (.bmp: 1) |
| **Resolution Range** | Min: $75 \times 75$ px \| Max: $6,410 \times 5,184$ px \| Mean: $1,015 \times 787$ px |

The audit identified 313 exact duplicate images within the dataset. Per explicit engineering guidelines, original files were left strictly intact, while duplicate occurrences were cataloged in the dataset inspection report.

---

## 3. Human Pose Detection & Keypoint Extraction (Phase 2)

### 3.1 MediaPipe Pose Landmarker Architecture
We integrated the latest **MediaPipe PoseLandmarker (Heavy float16 model)** (`models/pose_landmarker_heavy.task`), which provides high-precision localization of 33 body landmarks in complex acrobatic orientations.

Each landmark $i \in \{0, \dots, 32\}$ consists of:
- **Normalized coordinates**: $x_i \in [0, 1]$, $y_i \in [0, 1]$ (relative to image width and height)
- **Depth coordinate**: $z_i$ (representing depth relative to the midpoint of the hips)
- **Visibility score**: $v_i \in [0, 1]$ (probability that the landmark is within the camera frame)
- **Presence score**: $p_i \in [0, 1]$ (probability that the landmark exists on the human body)

### 3.2 Resilience and Detection Failure Handling
During processing, images containing alpha channels (PNG RGBA) or single-channel grayscale arrays initially triggered tensor dimension mismatches in LiteRT (`[224, 224, 3]` vs `[224, 224]`). To eliminate graph failures:
1. Implemented a custom safe loader converting all inputs into contiguous 3-channel RGB `uint8` arrays.
2. Built an automated detector recreation wrapper to reset the LiteRT execution graph if an unrecoverable C++ error occurs.

### 3.3 Detection Success & Failure Audit
Every processed image was tracked without silent discards. Unsuccessful detections were recorded in `results/dataset_analysis/detection_failures.csv`:

| Split | Total Processed | Successful Detections | Failures | Success Rate |
|---|:---:|:---:|:---:|:---:|
| **TRAIN** | 1,081 | 1,035 | 46 | **95.74%** |
| **TEST** | 470 | 464 | 6 | **98.72%** |
| **Overall** | 1,551 | 1,499 | 52 | **96.65%** |

---

## 4. Biomechanical Feature Engineering Methodology

To decouple classification from camera viewpoint, subject morphology, and image resolution, we engineered **202 structured features**:

### 4.1 Translation & Scale Normalization
1. **Mid-Hip Origin**:
   $$\mathbf{P}_{\text{mid\_hip}} = \frac{\mathbf{P}_{23} + \mathbf{P}_{24}}{2}$$
   All 33 landmark positions are centered relative to the pelvic midpoint:
   $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{mid\_hip}}$$

2. **Torso-Based Scale Normalization**:
   $$\mathbf{P}_{\text{mid\_shoulder}} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}$$
   $$S_{\text{torso}} = \|\mathbf{P}_{\text{mid\_shoulder}} - \mathbf{P}_{\text{mid\_hip}}\|_2$$
   Normalized coordinates:
   $$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{S_{\text{torso}}}$$

### 4.2 Biomechanical Joint Angles
For three consecutive joints $\mathbf{A}$, $\mathbf{B}$ (vertex), and $\mathbf{C}$, the 3D joint angle $\theta$ is:
$$\theta = \arccos\left(\text{clip}\left(\frac{(\mathbf{A} - \mathbf{B}) \cdot (\mathbf{C} - \mathbf{B})}{\|\mathbf{A} - \mathbf{B}\|_2 \|\mathbf{C} - \mathbf{B}\|_2}, -1.0, 1.0\right)\right) \times \frac{180^\circ}{\pi}$$

Key angles computed:
- **Elbow Angles (L/R)**: Wrist–Elbow–Shoulder
- **Knee Angles (L/R)**: Ankle–Knee–Hip
- **Hip Angles (L/R)**: Knee–Hip–Shoulder
- **Torso Vertical Angle**: Angle of $(\mathbf{P}_{\text{mid\_shoulder}} - \mathbf{P}_{\text{mid\_hip}})$ relative to gravity vector $(0, -1, 0)$

---

## 5. Machine Learning Model Development & Cross-Validation (Phase 4 & 5)

### 5-Fold Stratified CV Comparative Results (Training Set: 1,035 samples)

| Machine Learning Algorithm | Accuracy (Mean ± Std) | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **0.9478 ± 0.0168** | **0.9503 ± 0.0151** | **0.9451 ± 0.0179** | **0.9467 ± 0.0166** | **0.9477 ± 0.0168** |
| **Random Forest** | 0.9401 ± 0.0145 | 0.9458 ± 0.0132 | 0.9365 ± 0.0164 | 0.9394 ± 0.0149 | 0.9399 ± 0.0146 |
| **Support Vector Machine (RBF)** | 0.9324 ± 0.0097 | 0.9383 ± 0.0091 | 0.9302 ± 0.0105 | 0.9328 ± 0.0098 | 0.9324 ± 0.0097 |
| **K-Nearest Neighbors** | 0.9314 ± 0.0142 | 0.9358 ± 0.0125 | 0.9267 ± 0.0165 | 0.9301 ± 0.0147 | 0.9310 ± 0.0143 |
| **Decision Tree** | 0.9024 ± 0.0205 | 0.9016 ± 0.0210 | 0.9011 ± 0.0212 | 0.9004 ± 0.0211 | 0.9025 ± 0.0205 |

---

## 6. Final Model Training & Test Set Evaluation (Phase 6)

### 6.1 Overall Test Set Metrics

| Metric | Score | Percentage |
|---|:---:|:---:|
| **Test Accuracy** | **0.9957** | **99.57%** |
| **Balanced Accuracy** | **0.9954** | **99.54%** |
| **Macro Precision** | **0.9964** | **99.64%** |
| **Weighted Precision** | **0.9957** | **99.57%** |
| **Macro Recall** | **0.9954** | **99.54%** |
| **Weighted Recall** | **0.9957** | **99.57%** |
| **Macro F1-Score** | **0.9959** | **99.59%** |
| **Weighted F1-Score** | **0.9957** | **99.57%** |
| **Matthews Correlation Coefficient (MCC)** | **0.9946** | — |

### 6.2 Per-Class Detailed Performance

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **downdog** | 1.0000 | 0.9894 | 0.9947 | 94 |
| **goddess** | 1.0000 | 0.9875 | 0.9937 | 80 |
| **plank** | 0.9914 | 1.0000 | 0.9957 | 115 |
| **tree** | 1.0000 | 1.0000 | 1.0000 | 68 |
| **warrior2** | 0.9907 | 1.0000 | 0.9953 | 107 |
| **Macro Average** | **0.9964** | **0.9954** | **0.9959** | **464** |
| **Weighted Average** | **0.9957** | **0.9957** | **0.9957** | **464** |

### 6.3 Confusion Matrix (Test Set, N=464)

```
                       Predicted Class
               downdog  goddess  plank  tree  warrior2
      downdog       93        0      1     0         0
G     goddess        0       79      0     0         1
T       plank        0        0    115     0         0
         tree        0        0      0    68         0
     warrior2        0        0      0     0       107
```

---

## 7. Model Serialization & Standalone Prediction (Phase 7)

- Serialized model saved at: `models/best_yoga_pose_model.joblib`.
- Inference script: `src/predict.py`.

```powershell
python src/predict.py --image "DATASET/TEST/tree/00000000.jpg" --visualize "results/visualizations/sample_prediction_tree.jpg"
```
