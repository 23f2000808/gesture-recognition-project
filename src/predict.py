"""
Standalone Yoga Pose Prediction Pipeline
Phase 7: Inference on New Images Using Serialized Model Artifact
"""
import os
import sys
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import extract_features_from_landmarks, LANDMARK_NAMES

# MediaPipe skeleton connections for visualization
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12)
]

def load_image_rgb(image_path: Path):
    """Safely loads an image file and converts to 3-channel RGB uint8 format."""
    try:
        with open(image_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        return None, f"File reading error: {e}"

    if img is None:
        return None, f"cv2 could not decode image at {image_path}"

    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return None, f"Unsupported image shape: {img.shape}"

    return np.ascontiguousarray(img_rgb, dtype=np.uint8), None


def annotate_image(image_rgb: np.ndarray, landmarks, predicted_class: str, confidence: float):
    """Draws detected skeleton and prediction banner on the image."""
    annotated = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w, _ = annotated.shape

    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = (int(lm.x * w), int(lm.y * h))

    # Draw skeleton bones
    for p1, p2 in POSE_CONNECTIONS:
        cv2.line(annotated, pts[p1], pts[p2], (0, 255, 0), max(2, int(min(h, w) / 250)))

    # Draw landmark joints
    for i in range(33):
        cv2.circle(annotated, pts[i], max(3, int(min(h, w) / 180)), (0, 0, 255), -1)

    # Add overlay prediction banner at top
    banner_height = max(60, int(h * 0.1))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    text = f"Pose: {predicted_class.upper()}  |  Confidence: {confidence*100:.1f}%"
    font_scale = max(0.6, min(w, h) / 700.0)
    cv2.putText(annotated, text, (20, int(banner_height * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


def predict_pose(
    image_path: str,
    model_path: str = "models/best_yoga_pose_model.joblib",
    landmarker_model: str = "models/pose_landmarker_heavy.task",
    visualize_output: str = None
):
    """
    End-to-end prediction:
    1. Loads model artifact bundle
    2. Detects landmarks with MediaPipe PoseLandmarker
    3. Extracts identical normalized features and angles
    4. Runs classifier prediction and probability estimation
    5. Optionally saves annotated visualization
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mod_path = Path(model_path)
    if not mod_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Train the model first.")

    # 1. Load model bundle
    bundle = joblib.load(mod_path)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    classes = bundle["classes"]
    model_name = bundle.get("model_name", "Classifier")

    # 2. Load and preprocess image
    img_rgb, err = load_image_rgb(img_path)
    if img_rgb is None:
        return {"status": "error", "message": err}

    # 3. Detect pose with MediaPipe
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
        return {
            "status": "failure",
            "message": "No human body pose detected in image. Please ensure person is visible."
        }

    landmarks = detection_result.pose_landmarks[0]

    # 4. Extract features
    features = extract_features_from_landmarks(landmarks)
    df_features = pd.DataFrame([features])

    # Ensure all trained feature columns are present in exact order
    missing_cols = set(feature_names) - set(df_features.columns)
    for col in missing_cols:
        df_features[col] = 0.0
    df_features = df_features[feature_names]

    # 5. Predict class and probabilities
    pred_class = str(pipeline.predict(df_features)[0])
    
    probabilities = {}
    confidence = 1.0
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_features)[0]
        model_classes = pipeline.classes_
        for cls, p in zip(model_classes, probs):
            probabilities[str(cls)] = round(float(p), 4)
        confidence = float(np.max(probs))

    # 6. Optional visualization output
    vis_path_saved = None
    if visualize_output:
        annotated_bgr = annotate_image(img_rgb, landmarks, pred_class, confidence)
        out_vis = Path(visualize_output)
        out_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_vis), annotated_bgr)
        vis_path_saved = str(out_vis.resolve())

    return {
        "status": "success",
        "image": str(img_path.resolve()),
        "predicted_pose": pred_class,
        "confidence": round(confidence, 4),
        "class_probabilities": probabilities,
        "model_used": model_name,
        "annotated_visualization": vis_path_saved
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Yoga Pose from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input yoga image")
    parser.add_argument("--model", type=str, default="models/best_yoga_pose_model.joblib", help="Model joblib file")
    parser.add_argument("--landmarker-model", type=str, default="models/pose_landmarker_heavy.task", help="MediaPipe task file")
    parser.add_argument("--visualize", type=str, default=None, help="Path to save annotated output image")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    result = predict_pose(
        image_path=args.image,
        model_path=args.model,
        landmarker_model=args.landmarker_model,
        visualize_output=args.visualize
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 55)
        print("          YOGA POSE PREDICTION RESULT")
        print("=" * 55)
        if result["status"] == "success":
            print(f"  Input Image     : {Path(result['image']).name}")
            print(f"  PREDICTED POSE  : {result['predicted_pose'].upper()}")
            print(f"  Confidence      : {result['confidence']*100:.2f}%")
            print(f"  Model           : {result['model_used']}")
            print("\n  Class Probabilities:")
            for cls, prob in sorted(result["class_probabilities"].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"    - {cls:<10}: {prob*100:5.1f}%  {bar}")
            if result.get("annotated_visualization"):
                print(f"\n  Annotated Image : {result['annotated_visualization']}")
        else:
            print(f"  Status : {result['status'].upper()}")
            print(f"  Message: {result['message']}")
        print("=" * 55)

if __name__ == "__main__":
    main()
