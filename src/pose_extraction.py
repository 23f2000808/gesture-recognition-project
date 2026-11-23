"""
Robust Pose Extraction Pipeline using MediaPipe PoseLandmarker
Extracts 33 human pose landmarks and engineered features from yoga images.
Handles arbitrary color spaces, grayscale, alpha channels, and resilient detector recovery.
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Add src to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import extract_features_from_landmarks, LANDMARK_NAMES

def init_pose_landmarker(model_path: str = "models/pose_landmarker_heavy.task"):
    """Initializes a fresh MediaPipe PoseLandmarker detector."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pose landmarker model not found at {model_path}")
    
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options)


def load_image_as_rgb(image_path: Path):
    """
    Safely loads an image file using OpenCV and ensures standard 3-channel RGB uint8 format.
    Handles grayscale, RGBA, and unusual color formats.
    """
    # Use cv2.imdecode with raw bytes to handle non-ASCII paths safely on Windows
    try:
        with open(image_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        return None, f"File read error: {e}"

    if img is None:
        return None, "cv2 failed to decode image"

    if len(img.shape) == 2:
        # Grayscale -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # BGRA -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None, f"Unsupported channel count: {img.shape[2]}"
    else:
        return None, f"Unsupported image shape: {img.shape}"

    # Ensure contiguous uint8 array
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    return img_rgb, None


def process_image(detector, image_path: Path, model_path: str):
    """
    Detects pose landmarks from a single image and extracts engineered features.
    Recovers automatically if the detector graph enters an error state.
    
    Returns:
        detector, success (bool), feature_dict (dict or None), error_msg (str or None)
    """
    img_rgb, err = load_image_as_rgb(image_path)
    if img_rgb is None:
        return detector, False, None, err

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return detector, False, None, "No human pose detected"
            
        landmarks = detection_result.pose_landmarks[0]
        if len(landmarks) != 33:
            return detector, False, None, f"Incomplete landmarks ({len(landmarks)}/33)"
            
        features = extract_features_from_landmarks(landmarks)
        
        # Also store raw landmark coordinates for completeness
        for i, lm in enumerate(landmarks):
            name = LANDMARK_NAMES[i]
            features[f"{name}_raw_x"] = float(lm.x)
            features[f"{name}_raw_y"] = float(lm.y)
            features[f"{name}_raw_z"] = float(lm.z)
            
        return detector, True, features, None
    except Exception as e:
        err_str = str(e)
        # If the C++ graph failed, reinitialize detector so state resets
        if "Graph has errors" in err_str or "Calculator::Process" in err_str:
            try:
                detector.close()
            except Exception:
                pass
            detector = init_pose_landmarker(model_path)
        return detector, False, None, err_str


def extract_split(detector, split_dir: Path, split_name: str, model_path: str):
    """
    Processes all images in a dataset split (TRAIN or TEST).
    
    Returns:
        detector, records (list of dicts), failures (list of dicts)
    """
    records = []
    failures = []
    
    # Collect all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in split_dir.rglob("*") if p.suffix.lower() in valid_exts]
    print(f"\nProcessing {split_name} split: {len(image_paths)} images found in {split_dir}")
    
    for img_path in tqdm(sorted(image_paths), desc=f"Extracting {split_name}"):
        class_name = img_path.parent.name
        img_name = img_path.name
        
        detector, success, features, err = process_image(detector, img_path, model_path)
        
        record = {
            "image_name": img_name,
            "image_path": str(img_path),
            "split": split_name,
            "label": class_name,
            "detection_success": success
        }
        
        if success and features is not None:
            record.update(features)
            records.append(record)
        else:
            failures.append({
                "image_name": img_name,
                "image_path": str(img_path),
                "split": split_name,
                "label": class_name,
                "error": err
            })
            record["detection_error"] = err
            records.append(record)
            
    return detector, records, failures


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose Extraction Pipeline")
    parser.add_argument("--dataset-dir", type=str, default="DATASET", help="Root directory of dataset")
    parser.add_argument("--model-path", type=str, default="models/pose_landmarker_heavy.task", help="MediaPipe task file")
    parser.add_argument("--output-dir", type=str, default="features", help="Output directory for features")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / args.dataset_dir
    model_path = project_root / args.model_path
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = project_root / "results" / "dataset_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_out_csv = output_dir / "train_features.csv"
    test_out_csv = output_dir / "test_features.csv"
    failures_csv = results_dir / "detection_failures.csv"

    print("=" * 60)
    print("PHASE 2: Human Pose Detection and Keypoint Extraction")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print("=" * 60)

    detector = init_pose_landmarker(str(model_path))

    all_failures = []

    # Process TRAIN
    train_dir = dataset_dir / "TRAIN"
    if not train_dir.exists():
        train_dir = dataset_dir / "train"
    detector, train_records, train_failures = extract_split(detector, train_dir, "TRAIN", str(model_path))
    all_failures.extend(train_failures)

    # Process TEST
    test_dir = dataset_dir / "TEST"
    if not test_dir.exists():
        test_dir = dataset_dir / "test"
    detector, test_records, test_failures = extract_split(detector, test_dir, "TEST", str(model_path))
    all_failures.extend(test_failures)

    # Save to CSV
    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    failures_df = pd.DataFrame(all_failures)

    train_df.to_csv(train_out_csv, index=False)
    test_df.to_csv(test_out_csv, index=False)
    failures_df.to_csv(failures_csv, index=False)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"TRAIN Set:")
    print(f"  Total Images    : {len(train_records)}")
    print(f"  Successes       : {train_df['detection_success'].sum()} ({train_df['detection_success'].mean()*100:.1f}%)")
    print(f"  Failures        : {len(train_failures)}")
    print(f"  Saved to        : {train_out_csv}")

    print(f"\nTEST Set:")
    print(f"  Total Images    : {len(test_records)}")
    print(f"  Successes       : {test_df['detection_success'].sum()} ({test_df['detection_success'].mean()*100:.1f}%)")
    print(f"  Failures        : {len(test_failures)}")
    print(f"  Saved to        : {test_out_csv}")

    print(f"\nTotal Failures Recorded: {len(all_failures)}")
    print(f"Failures saved to      : {failures_csv}")

    # Breakdown by class
    print("\nDetection Success Rate by Class (TRAIN):")
    train_rates = train_df.groupby("label")["detection_success"].agg(["count", "sum", "mean"])
    train_rates.columns = ["Total", "Detected", "Success Rate"]
    train_rates["Success Rate"] = (train_rates["Success Rate"] * 100).round(1).astype(str) + "%"
    print(train_rates)

    print("\nDetection Success Rate by Class (TEST):")
    test_rates = test_df.groupby("label")["detection_success"].agg(["count", "sum", "mean"])
    test_rates.columns = ["Total", "Detected", "Success Rate"]
    test_rates["Success Rate"] = (test_rates["Success Rate"] * 100).round(1).astype(str) + "%"
    print(test_rates)

if __name__ == "__main__":
    main()
