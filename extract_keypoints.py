import csv
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton

# Ensure weights load from the local project models directory
os.environ['TORCH_HOME'] = str(Path(__file__).resolve().parent / 'models')

# Mapping for the 17 standard COCO body keypoints from the 133 Wholebody landmarks
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def init_pose_estimator():
    """Initializes the RTMLib Wholebody pose estimator using lightweight CPU models."""
    return Wholebody(
        mode='lightweight',
        backend='onnxruntime',
        device='cpu'
    )


def extract_keypoints_single_image(estimator, image_path: str):
    """
    Extracts keypoints and scores for a single image.
    
    Returns:
        keypoints: np.ndarray of shape (num_persons, 133, 2)
        scores: np.ndarray of shape (num_persons, 133)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    keypoints, scores = estimator(img)
    return keypoints, scores, img


def extract_dataset(dataset_dir: str = "DATASET", output_csv: str = "yoga_keypoints.csv"):
    """
    Iterates through dataset directories (TRAIN/TEST and class subfolders),
    extracts the primary person's body keypoints, and saves them to a CSV.
    """
    estimator = init_pose_estimator()
    dataset_path = Path(dataset_dir)
    
    # Define columns: split, label, filename, followed by (x, y, score) for each landmark
    columns = ["split", "label", "image_name"]
    for kpt in COCO_KEYPOINT_NAMES:
        columns.extend([f"{kpt}_x", f"{kpt}_y", f"{kpt}_score"])
    
    rows = []
    image_paths = list(dataset_path.glob("*/*/*.jpg"))
    print(f"Found {len(image_paths)} images across {dataset_dir}")

    for img_path in tqdm(image_paths, desc="Extracting Keypoints"):
        split = img_path.parent.parent.name  # TRAIN or TEST
        label = img_path.parent.name         # downdog, goddess, plank, tree, warrior2
        
        try:
            keypoints, scores, _ = extract_keypoints_single_image(estimator, str(img_path))
            
            # If at least one person is detected, take the highest-confidence person (index 0)
            if len(keypoints) > 0:
                kpts_17 = keypoints[0][:17]  # First 17 keypoints correspond to COCO body
                scores_17 = scores[0][:17]
                
                row = [split, label, img_path.name]
                for (x, y), score in zip(kpts_17, scores_17):
                    row.extend([round(float(x), 2), round(float(y), 2), round(float(score), 4)])
                rows.append(row)
            else:
                # No person detected; fill with NaNs
                row = [split, label, img_path.name] + [np.nan] * (len(COCO_KEYPOINT_NAMES) * 3)
                rows.append(row)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Extraction complete! Saved {len(rows)} records to {output_csv}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract keypoints from Yoga Dataset images")
    parser.add_argument("--demo-image", type=str, help="Path to single image to test and visualize")
    parser.add_argument("--extract-all", action="store_true", help="Extract keypoints for the entire dataset")
    parser.add_argument("--output", type=str, default="yoga_keypoints.csv", help="Output CSV filename")
    args = parser.parse_args()

    if args.demo_image:
        estimator = init_pose_estimator()
        kpts, scs, img = extract_keypoints_single_image(estimator, args.demo_image)
        print(f"Detected {len(kpts)} person(s)")
        if len(kpts) > 0:
            print(f"Body keypoints shape: {kpts.shape}")
            annotated = draw_skeleton(img, kpts, scs, kpt_thr=0.4)
            out_preview = "preview_pose.jpg"
            cv2.imwrite(out_preview, annotated)
            print(f"Saved visualization preview to {out_preview}")
    elif args.extract_all:
        extract_dataset(dataset_dir="DATASET", output_csv=args.output)
    else:
        print("Usage:")
        print("  python extract_keypoints.py --demo-image DATASET/TRAIN/tree/00000070.jpg")
        print("  python extract_keypoints.py --extract-all --output yoga_keypoints.csv")