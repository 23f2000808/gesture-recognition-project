"""
Feature Engineering for Yoga Pose Recognition
Extracts normalized body-centric coordinates, joint angles, and biomechanical distance ratios.
"""
import numpy as np
from typing import List, Dict, Any, Optional

# MediaPipe Pose Landmark Indices (33 landmarks)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the 3D angle (in degrees) at vertex b between vectors ba and bc.
    a, b, c are 3D numpy arrays [x, y, z].
    """
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cosine)))
    return angle_deg


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the 2D angle (in degrees) in image plane at vertex b between ba and bc.
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_features_from_landmarks(landmarks: List[Any]) -> Dict[str, float]:
    """
    Given a list of 33 MediaPipe pose landmark objects (each having x, y, z, visibility, presence),
    computes:
      1. Body-centered and scale-normalized coordinates (x, y, z) for all 33 landmarks.
      2. Landmark visibility and presence scores.
      3. Biomechanical joint angles (elbows, shoulders, hips, knees, ankles, spine, inter-limb).
      4. Scale-invariant distance ratios and relative vertical positioning.

    Returns a flat dictionary mapping feature_name -> feature_value.
    """
    if len(landmarks) != 33:
        raise ValueError(f"Expected 33 landmarks, got {len(landmarks)}")

    # Convert to numpy arrays: shape (33, 3) for coordinates
    coords = np.zeros((33, 3), dtype=np.float32)
    visibilities = np.zeros(33, dtype=np.float32)
    presences = np.zeros(33, dtype=np.float32)

    for i, lm in enumerate(landmarks):
        coords[i] = [lm.x, lm.y, lm.z]
        visibilities[i] = getattr(lm, "visibility", 1.0) if getattr(lm, "visibility", None) is not None else 1.0
        presences[i] = getattr(lm, "presence", 1.0) if getattr(lm, "presence", None) is not None else 1.0

    features = {}

    # Key landmark indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW       = 13, 14
    L_WRIST, R_WRIST       = 15, 16
    L_HIP, R_HIP           = 23, 24
    L_KNEE, R_KNEE         = 25, 26
    L_ANKLE, R_ANKLE       = 27, 28
    L_FOOT, R_FOOT         = 31, 32
    NOSE                   = 0

    # 1. Body reference points: Mid-Hip and Mid-Shoulder
    mid_hip = (coords[L_HIP] + coords[R_HIP]) / 2.0
    mid_shoulder = (coords[L_SHOULDER] + coords[R_SHOULDER]) / 2.0
    
    # Scale reference: Torso size (distance from mid-shoulder to mid-hip)
    torso_size = np.linalg.norm(mid_shoulder - mid_hip)
    if torso_size < 1e-4:
        # Fallback to shoulder width or bbox dimension
        shoulder_width = np.linalg.norm(coords[L_SHOULDER] - coords[R_SHOULDER])
        torso_size = shoulder_width if shoulder_width > 1e-4 else 1.0

    # 2. Normalized coordinates relative to mid-hip origin and torso scale
    norm_coords = (coords - mid_hip) / torso_size

    for i, name in enumerate(LANDMARK_NAMES):
        features[f"{name}_norm_x"] = float(norm_coords[i, 0])
        features[f"{name}_norm_y"] = float(norm_coords[i, 1])
        features[f"{name}_norm_z"] = float(norm_coords[i, 2])
        features[f"{name}_vis"]    = float(visibilities[i])
        features[f"{name}_pres"]   = float(presences[i])

    # 3. Biomechanical Joint Angles (3D)
    # Arms
    features["angle_left_elbow_3d"]   = calculate_angle_3d(coords[L_SHOULDER], coords[L_ELBOW], coords[L_WRIST])
    features["angle_right_elbow_3d"]  = calculate_angle_3d(coords[R_SHOULDER], coords[R_ELBOW], coords[R_WRIST])
    features["angle_left_shoulder_3d"]= calculate_angle_3d(coords[L_ELBOW], coords[L_SHOULDER], coords[L_HIP])
    features["angle_right_shoulder_3d"]= calculate_angle_3d(coords[R_ELBOW], coords[R_SHOULDER], coords[R_HIP])
    
    # Legs
    features["angle_left_hip_3d"]     = calculate_angle_3d(coords[L_SHOULDER], coords[L_HIP], coords[L_KNEE])
    features["angle_right_hip_3d"]    = calculate_angle_3d(coords[R_SHOULDER], coords[R_HIP], coords[R_KNEE])
    features["angle_left_knee_3d"]    = calculate_angle_3d(coords[L_HIP], coords[L_KNEE], coords[L_ANKLE])
    features["angle_right_knee_3d"]   = calculate_angle_3d(coords[R_HIP], coords[R_KNEE], coords[R_ANKLE])
    features["angle_left_ankle_3d"]   = calculate_angle_3d(coords[L_KNEE], coords[L_ANKLE], coords[L_FOOT])
    features["angle_right_ankle_3d"]  = calculate_angle_3d(coords[R_KNEE], coords[R_ANKLE], coords[R_FOOT])

    # 2D Joint Angles (robust to depth estimation noise)
    features["angle_left_elbow_2d"]   = calculate_angle_2d(coords[L_SHOULDER], coords[L_ELBOW], coords[L_WRIST])
    features["angle_right_elbow_2d"]  = calculate_angle_2d(coords[R_SHOULDER], coords[R_ELBOW], coords[R_WRIST])
    features["angle_left_shoulder_2d"]= calculate_angle_2d(coords[L_ELBOW], coords[L_SHOULDER], coords[L_HIP])
    features["angle_right_shoulder_2d"]= calculate_angle_2d(coords[R_ELBOW], coords[R_SHOULDER], coords[R_HIP])
    features["angle_left_hip_2d"]     = calculate_angle_2d(coords[L_SHOULDER], coords[L_HIP], coords[L_KNEE])
    features["angle_right_hip_2d"]    = calculate_angle_2d(coords[R_SHOULDER], coords[R_HIP], coords[R_KNEE])
    features["angle_left_knee_2d"]    = calculate_angle_2d(coords[L_HIP], coords[L_KNEE], coords[L_ANKLE])
    features["angle_right_knee_2d"]   = calculate_angle_2d(coords[R_HIP], coords[R_KNEE], coords[R_ANKLE])
    features["angle_left_ankle_2d"]   = calculate_angle_2d(coords[L_KNEE], coords[L_ANKLE], coords[L_FOOT])
    features["angle_right_ankle_2d"]  = calculate_angle_2d(coords[R_KNEE], coords[R_ANKLE], coords[R_FOOT])

    # Torso & Spine
    # Spine angle with vertical: vector mid_hip -> mid_shoulder compared to vertical vector (0, -1, 0)
    spine_vec = mid_shoulder - mid_hip
    vertical_vec = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    spine_norm = np.linalg.norm(spine_vec)
    if spine_norm > 1e-4:
        cos_spine = np.dot(spine_vec[:2], vertical_vec[:2]) / (np.linalg.norm(spine_vec[:2]) + 1e-6)
        features["angle_torso_vertical_deg"] = float(np.degrees(np.arccos(np.clip(cos_spine, -1.0, 1.0))))
    else:
        features["angle_torso_vertical_deg"] = 0.0

    # Inter-limb spread angles
    features["angle_inter_arm_spread"] = calculate_angle_3d(coords[L_WRIST], mid_shoulder, coords[R_WRIST])
    features["angle_inter_leg_spread"] = calculate_angle_3d(coords[L_ANKLE], mid_hip, coords[R_ANKLE])

    # 4. Normalized Distance Ratios
    features["dist_wrist_to_wrist_ratio"] = float(np.linalg.norm(coords[L_WRIST] - coords[R_WRIST]) / torso_size)
    features["dist_ankle_to_ankle_ratio"] = float(np.linalg.norm(coords[L_ANKLE] - coords[R_ANKLE]) / torso_size)
    features["dist_left_wrist_to_hip_ratio"]  = float(np.linalg.norm(coords[L_WRIST] - coords[L_HIP]) / torso_size)
    features["dist_right_wrist_to_hip_ratio"] = float(np.linalg.norm(coords[R_WRIST] - coords[R_HIP]) / torso_size)
    features["dist_left_ankle_to_hip_ratio"]  = float(np.linalg.norm(coords[L_ANKLE] - coords[L_HIP]) / torso_size)
    features["dist_right_ankle_to_hip_ratio"] = float(np.linalg.norm(coords[R_ANKLE] - coords[R_HIP]) / torso_size)
    
    shoulder_dist = np.linalg.norm(coords[L_SHOULDER] - coords[R_SHOULDER])
    hip_dist = np.linalg.norm(coords[L_HIP] - coords[R_HIP])
    features["ratio_shoulder_to_hip_width"] = float(shoulder_dist / (hip_dist + 1e-6))

    # 5. Relative Vertical Positions (y increases downwards in image coordinates)
    # Head above or below hips: negative means head is higher (above) hips, positive means inverted (downdog)
    features["rel_head_to_hip_y"] = float((coords[NOSE, 1] - mid_hip[1]) / torso_size)
    features["rel_left_wrist_to_shoulder_y"] = float((coords[L_WRIST, 1] - coords[L_SHOULDER, 1]) / torso_size)
    features["rel_right_wrist_to_shoulder_y"] = float((coords[R_WRIST, 1] - coords[R_SHOULDER, 1]) / torso_size)
    features["rel_left_ankle_to_knee_y"] = float((coords[L_ANKLE, 1] - coords[L_KNEE, 1]) / torso_size)
    features["rel_right_ankle_to_knee_y"] = float((coords[R_ANKLE, 1] - coords[R_KNEE, 1]) / torso_size)

    # 6. Overall Body Bounding Box Aspect Ratio (width / height)
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    features["bbox_aspect_ratio"] = float(bbox_w / (bbox_h + 1e-6))
    features["torso_size"] = float(torso_size)

    return features
