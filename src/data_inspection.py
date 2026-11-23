"""
Data Inspection Script for Yoga Pose Recognition
Phase 1: Project and Dataset Inspection
"""
import os
import sys
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image

def check_installed_libraries():
    """Verify that all required libraries can be imported and report versions."""
    required = [
        "mediapipe",
        "cv2",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "PIL"
    ]
    status = {}
    for pkg in required:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            status[pkg] = {"installed": True, "version": str(version)}
        except ImportError as e:
            status[pkg] = {"installed": False, "error": str(e)}
    return status


def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Calculate MD5 hash of a file to detect duplicates."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_dataset(dataset_root: Path):
    """
    Thoroughly inspect the dataset:
    - Counts per split and class
    - File formats
    - Image readability with OpenCV and PIL
    - Image dimensions (min, max, mean)
    - Corrupted / unreadable / truncated files
    - Duplicate files
    """
    results = {
        "dataset_root": str(dataset_root.resolve()),
        "splits": {},
        "corrupted_images": [],
        "duplicate_images": [],
        "unreadable_cv2": [],
        "total_images": 0,
        "classes": set(),
        "formats": defaultdict(int),
        "resolutions": []
    }

    seen_hashes = {} # hash -> filepath

    for split in ["TRAIN", "TEST"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            # Try lowercase
            split_dir = dataset_root / split.lower()
            if not split_dir.exists():
                print(f"Warning: Split {split} not found in {dataset_root}")
                continue

        split_name = split.upper()
        results["splits"][split_name] = {
            "classes": defaultdict(int),
            "total": 0
        }

        # Subdirectories are class folders
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        for cdir in sorted(class_dirs, key=lambda d: d.name):
            class_name = cdir.name
            results["classes"].add(class_name)
            
            image_files = [f for f in cdir.iterdir() if f.is_file()]
            for img_file in image_files:
                results["total_images"] += 1
                results["splits"][split_name]["classes"][class_name] += 1
                results["splits"][split_name]["total"] += 1
                ext = img_file.suffix.lower()
                results["formats"][ext] += 1

                # 1. Check duplicate
                try:
                    f_hash = compute_file_hash(img_file)
                    if f_hash in seen_hashes:
                        results["duplicate_images"].append({
                            "original": str(seen_hashes[f_hash]),
                            "duplicate": str(img_file),
                            "split": split_name,
                            "class": class_name
                        })
                    else:
                        seen_hashes[f_hash] = img_file
                except Exception as e:
                    results["corrupted_images"].append({
                        "file": str(img_file),
                        "reason": f"Hash computation error: {e}"
                    })
                    continue

                # 2. Check PIL verification (corrupted headers/truncated data)
                try:
                    with Image.open(img_file) as pil_img:
                        pil_img.verify()
                except Exception as e:
                    results["corrupted_images"].append({
                        "file": str(img_file),
                        "reason": f"PIL verify failed: {e}"
                    })

                # 3. Check OpenCV reading
                try:
                    cv_img = cv2.imread(str(img_file))
                    if cv_img is None:
                        results["unreadable_cv2"].append({
                            "file": str(img_file),
                            "reason": "cv2.imread returned None"
                        })
                    else:
                        h, w, c = cv_img.shape
                        results["resolutions"].append((w, h))
                except Exception as e:
                    results["unreadable_cv2"].append({
                        "file": str(img_file),
                        "reason": f"cv2.imread exception: {e}"
                    })

    # Convert sets/defaultdicts to regular structures for JSON serialization
    results["classes"] = sorted(list(results["classes"]))
    for split_name in results["splits"]:
        results["splits"][split_name]["classes"] = dict(results["splits"][split_name]["classes"])
    results["formats"] = dict(results["formats"])

    return results


def main():
    print("=" * 60)
    print("PHASE 1: Project and Dataset Inspection")
    print("=" * 60)

    # 1. Environment check
    print("\n1. Checking Python Environment & Dependencies...")
    lib_status = check_installed_libraries()
    for lib, info in lib_status.items():
        status_str = f"OK (v{info['version']})" if info["installed"] else f"MISSING ({info.get('error')})"
        print(f"  - {lib:<15}: {status_str}")

    # 2. Dataset path discovery
    project_root = Path(__file__).resolve().parent.parent
    dataset_candidates = [
        project_root / "DATASET",
        project_root / "dataset",
        project_root / "data"
    ]
    dataset_path = None
    for cand in dataset_candidates:
        if cand.exists() and cand.is_dir():
            dataset_path = cand
            break

    if not dataset_path:
        print(f"\nERROR: Dataset directory not found in candidates: {[str(c) for c in dataset_candidates]}")
        sys.exit(1)

    print(f"\n2. Discovered Dataset Path: {dataset_path.resolve()}")

    # 3. Inspect dataset
    print("\n3. Inspecting images, file formats, and class balance...")
    results = inspect_dataset(dataset_path)

    # Calculate resolution stats
    if results["resolutions"]:
        widths = [r[0] for r in results["resolutions"]]
        heights = [r[1] for r in results["resolutions"]]
        res_summary = {
            "min_width": min(widths), "max_width": max(widths), "avg_width": round(sum(widths)/len(widths), 1),
            "min_height": min(heights), "max_height": max(heights), "avg_height": round(sum(heights)/len(heights), 1),
        }
    else:
        res_summary = {}

    print(f"\nTotal Images Found: {results['total_images']}")
    print(f"Classes ({len(results['classes'])}): {', '.join(results['classes'])}")
    print(f"File Formats: {results['formats']}")
    print(f"Resolution stats: {res_summary}")

    print("\nSplit & Class Distribution:")
    for split, data in results["splits"].items():
        print(f"  [{split}] Total = {data['total']}")
        for cls, count in data["classes"].items():
            pct = (count / data['total'] * 100) if data['total'] > 0 else 0
            print(f"    - {cls:<12}: {count:>4} images ({pct:5.1f}%)")

    print(f"\nData Quality Issues:")
    print(f"  - Corrupted Images (PIL verify): {len(results['corrupted_images'])}")
    print(f"  - Unreadable Images (OpenCV)   : {len(results['unreadable_cv2'])}")
    print(f"  - Duplicate Images (Exact MD5) : {len(results['duplicate_images'])}")

    if results["duplicate_images"]:
        print("\nSample Duplicate Images:")
        for dup in results["duplicate_images"][:5]:
            print(f"    Dup: {Path(dup['duplicate']).name} matches {Path(dup['original']).name} (Class: {dup['class']}, Split: {dup['split']})")

    # Output detailed JSON report
    out_dir = project_root / "results" / "dataset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "dataset_inspection_report.json"
    
    # Exclude raw resolution list to keep report compact
    summary_to_save = {
        "environment": lib_status,
        "dataset_root": results["dataset_root"],
        "total_images": results["total_images"],
        "classes": results["classes"],
        "formats": results["formats"],
        "resolution_summary": res_summary,
        "splits": results["splits"],
        "duplicate_count": len(results["duplicate_images"]),
        "duplicate_samples": results["duplicate_images"][:20],
        "corrupted_count": len(results["corrupted_images"]),
        "corrupted_samples": results["corrupted_images"],
        "unreadable_cv2_count": len(results["unreadable_cv2"]),
        "unreadable_cv2_samples": results["unreadable_cv2"],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_to_save, f, indent=2)
    print(f"\nDetailed inspection report saved to: {report_file}")

if __name__ == "__main__":
    main()
