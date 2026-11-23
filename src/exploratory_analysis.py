"""
Exploratory Data Analysis (EDA) for Yoga Pose Recognition
Generates comprehensive visualizations of keypoints, skeletons, class distributions,
joint angle distributions, failure analysis, and feature correlations.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MediaPipe skeletal connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Head / Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]

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


def set_plot_style():
    """Sets a clean, professional aesthetic for scientific visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = 300


def plot_class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path):
    """Visualizes class distribution in Train and Test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = sns.color_palette("Set2")

    train_counts = train_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()

    sns.barplot(x=train_counts.index, y=train_counts.values, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Training Set Class Distribution", fontweight="bold")
    axes[0].set_xlabel("Yoga Pose Class")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v + 3, f"{v}\n({v/len(train_df)*100:.1f}%)", ha="center", fontsize=10)

    sns.barplot(x=test_counts.index, y=test_counts.values, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Testing Set Class Distribution", fontweight="bold")
    axes[1].set_xlabel("Yoga Pose Class")
    for i, v in enumerate(test_counts.values):
        axes[1].text(i, v + 3, f"{v}\n({v/len(test_df)*100:.1f}%)", ha="center", fontsize=10)

    plt.tight_layout()
    out_file = out_dir / "class_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_detection_failures(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path):
    """Visualizes detection success and failure rate per class and split."""
    combined = pd.concat([
        train_df[["label", "split", "detection_success"]],
        test_df[["label", "split", "detection_success"]]
    ])
    
    summary = combined.groupby(["split", "label"])["detection_success"].agg(
        Total="count",
        Success="sum"
    ).reset_index()
    summary["Failure"] = summary["Total"] - summary["Success"]
    summary["Failure_Rate_Pct"] = (summary["Failure"] / summary["Total"]) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="label", y="Failure_Rate_Pct", hue="split", palette="magma", ax=ax)
    ax.set_title("Pose Detection Failure Rate by Class and Split (%)", fontweight="bold")
    ax.set_xlabel("Yoga Pose Class")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_ylim(0, max(15, summary["Failure_Rate_Pct"].max() + 5))

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 0.3),
                        ha="center", fontsize=9)

    plt.tight_layout()
    out_file = out_dir / "detection_failure_analysis.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_canonical_skeletons(train_df: pd.DataFrame, out_dir: Path):
    """
    Computes and plots the mean normalized skeletal shape for each yoga pose.
    Demonstrates the structural separation of poses in normalized coordinate space.
    """
    valid_df = train_df[train_df["detection_success"] == True].copy()
    classes = sorted(valid_df["label"].unique())

    fig, axes = plt.subplots(1, len(classes), figsize=(18, 4), sharex=True, sharey=True)
    
    for idx, cls in enumerate(classes):
        cls_df = valid_df[valid_df["label"] == cls]
        ax = axes[idx]
        
        # Calculate mean normalized x and y for each landmark
        mean_coords = np.zeros((33, 2))
        for i, name in enumerate(LANDMARK_NAMES):
            mean_coords[i, 0] = cls_df[f"{name}_norm_x"].mean()
            # MediaPipe y increases downwards; invert for standard Cartesian plotting
            mean_coords[i, 1] = -cls_df[f"{name}_norm_y"].mean()

        # Draw bones
        for start_idx, end_idx in POSE_CONNECTIONS:
            # Differentiate bone groups by color
            if start_idx in [11, 12, 23, 24] and end_idx in [11, 12, 23, 24]:
                color = "#2b5c8f" # Torso - blue
                lw = 2.5
            elif start_idx in [11, 13, 15, 17, 19, 21] or end_idx in [11, 13, 15, 17, 19, 21]:
                color = "#e66101" # Left arm - orange
                lw = 2.0
            elif start_idx in [12, 14, 16, 18, 20, 22] or end_idx in [12, 14, 16, 18, 20, 22]:
                color = "#b2182b" # Right arm - red
                lw = 2.0
            elif start_idx in [23, 25, 27, 29, 31] or end_idx in [23, 25, 27, 29, 31]:
                color = "#4dac26" # Left leg - green
                lw = 2.0
            elif start_idx in [24, 26, 28, 30, 32] or end_idx in [24, 26, 28, 30, 32]:
                color = "#018571" # Right leg - teal
                lw = 2.0
            else:
                color = "#7b3294" # Head - purple
                lw = 1.5

            ax.plot([mean_coords[start_idx, 0], mean_coords[end_idx, 0]],
                    [mean_coords[start_idx, 1], mean_coords[end_idx, 1]],
                    color=color, linewidth=lw, alpha=0.9)

        # Draw joint nodes
        ax.scatter(mean_coords[:, 0], mean_coords[:, 1], color="#2c3e50", s=30, zorder=5)
        
        ax.set_title(f"{cls.upper()}\n(N={len(cls_df)})", fontweight="bold", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.0, 2.0)

    fig.suptitle("Canonical Mean Normalized Skeletons Across Yoga Poses", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_file = out_dir / "canonical_mean_skeletons.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_joint_angles_distribution(train_df: pd.DataFrame, out_dir: Path):
    """Compares key joint angles across the 5 yoga poses using boxplots."""
    valid_df = train_df[train_df["detection_success"] == True].copy()
    
    angles_to_plot = [
        ("angle_left_knee_3d", "Left Knee Angle (°)"),
        ("angle_right_knee_3d", "Right Knee Angle (°)"),
        ("angle_left_hip_3d", "Left Hip Angle (°)"),
        ("angle_torso_vertical_deg", "Torso Inclination to Vertical (°)"),
        ("angle_left_elbow_3d", "Left Elbow Angle (°)"),
        ("angle_inter_leg_spread", "Inter-Leg Spread Angle (°)")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(angles_to_plot):
        ax = axes[idx]
        sns.boxplot(data=valid_df, x="label", y=col, ax=ax, palette="Set3", showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"5"})
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Degrees (°)")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Biomechanical Joint Angle Distributions Across Yoga Poses", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_file = out_dir / "joint_angles_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_correlation(train_df: pd.DataFrame, out_dir: Path):
    """Computes and visualizes correlation matrix for key engineered geometric features."""
    valid_df = train_df[train_df["detection_success"] == True].copy()
    
    key_features = [
        "angle_left_elbow_3d", "angle_right_elbow_3d",
        "angle_left_shoulder_3d", "angle_right_shoulder_3d",
        "angle_left_hip_3d", "angle_right_hip_3d",
        "angle_left_knee_3d", "angle_right_knee_3d",
        "angle_torso_vertical_deg", "angle_inter_leg_spread",
        "dist_wrist_to_wrist_ratio", "dist_ankle_to_ankle_ratio",
        "rel_head_to_hip_y", "bbox_aspect_ratio"
    ]
    
    feature_names_clean = [
        "L-Elbow Angle", "R-Elbow Angle",
        "L-Shoulder Angle", "R-Shoulder Angle",
        "L-Hip Angle", "R-Hip Angle",
        "L-Knee Angle", "R-Knee Angle",
        "Torso Inclination", "Leg Spread",
        "Wrist-Wrist Dist", "Ankle-Ankle Dist",
        "Head-Hip Y", "Aspect Ratio"
    ]

    corr = valid_df[key_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=feature_names_clean, yticklabels=feature_names_clean,
                cbar_kws={"label": "Pearson Correlation Coefficient"},
                linewidths=0.5)
    plt.title("Correlation Heatmap of Key Engineered Biomechanical Features", fontweight="bold", pad=15)
    plt.tight_layout()
    out_file = out_dir / "feature_correlation_heatmap.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_sample_poses_with_skeleton(train_df: pd.DataFrame, out_dir: Path):
    """
    Overlays detected MediaPipe landmarks and skeletal connections on 1 sample image per class.
    """
    valid_df = train_df[train_df["detection_success"] == True].copy()
    classes = sorted(valid_df["label"].unique())
    
    fig, axes = plt.subplots(1, len(classes), figsize=(20, 5))

    for idx, cls in enumerate(classes):
        sample_row = valid_df[valid_df["label"] == cls].iloc[0]
        img_path = sample_row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Landmark pixel coordinates
        pts = {}
        for i, name in enumerate(LANDMARK_NAMES):
            raw_x = sample_row[f"{name}_raw_x"]
            raw_y = sample_row[f"{name}_raw_y"]
            pts[i] = (int(raw_x * w), int(raw_y * h))

        # Draw bones
        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(img_rgb, pts[start_idx], pts[end_idx], (0, 255, 0), max(2, int(min(h, w) / 200)))

        # Draw joints
        for i in range(33):
            cv2.circle(img_rgb, pts[i], max(3, int(min(h, w) / 150)), (255, 0, 0), -1)

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{cls.upper()}\n({sample_row['image_name']})", fontweight="bold")
        axes[idx].axis("off")

    fig.suptitle("MediaPipe 33-Landmark Pose Estimation on Sample Yoga Poses", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_file = out_dir / "sample_pose_landmarks_overlay.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    set_plot_style()
    project_root = Path(__file__).resolve().parent.parent
    train_csv = project_root / "features" / "train_features.csv"
    test_csv = project_root / "features" / "test_features.csv"
    out_dir = project_root / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not test_csv.exists():
        print(f"Error: Feature CSV files not found. Run pose_extraction.py first.")
        sys.exit(1)

    print("Loading feature datasets...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded {len(train_df)} TRAIN and {len(test_df)} TEST records.")
    print("Generating exploratory visualizations...")

    plot_class_distribution(train_df, test_df, out_dir)
    plot_detection_failures(train_df, test_df, out_dir)
    plot_canonical_skeletons(train_df, out_dir)
    plot_joint_angles_distribution(train_df, out_dir)
    plot_feature_correlation(train_df, out_dir)
    plot_sample_poses_with_skeleton(train_df, out_dir)

    print("\nEDA Visualizations successfully generated and saved to:")
    print(f"  {out_dir}")

if __name__ == "__main__":
    main()
