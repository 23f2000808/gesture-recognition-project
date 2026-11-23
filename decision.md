# Decision log

Each entry is a choice made while setting up pose estimation for this project, and why it was made. Later changes should be appended, not rewritten over earlier entries.

## 2026-10-08 — Keep installs inside this project   

**Decision:** Create a virtual environment at `.venv` and put model files under `models/`. Do not install packages into the system Python.

**Reason:** The environment has to stay local to this repository.

**How to use it:**

```powershell
.\.venv\Scripts\Activate.ps1
$env:TORCH_HOME = "D:\gesture-recognition-project\models"
```

`TORCH_HOME` must be set before importing `rtmlib`. Without it, RTMLib downloads weights to the user cache (`~/.cache/rtmlib`) instead of this project.

## 2026-10-08 — Use Python 3.11.9 for the virtual environment

**Decision:** Build `.venv` with `py -3.11` (Python 3.11.9). Leave the system default, Python 3.14.2 at `C:\Python314\python.exe`, unused.

**Reason:** 3.14 is too new for the current ONNX Runtime and OpenCV wheels this stack needs. 3.11.9 is already installed and has those wheels.

## 2026-10-08 — Drop OpenPose

**Decision:** Do not install CMU OpenPose. Use RTMLib instead.

**What was tried:**

- `pip index` for `openpose`, `pyopenpose`, and `openpose-python` inside `.venv`. All three returned `No matching distribution found`. There is no pip package for CMU OpenPose.

**Obstacles that stopped a source build:**

- CMake is not installed (`cmake` is not on PATH).
- The CUDA toolkit is not installed (`nvcc` is missing, and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` does not exist). A GPU OpenPose build cannot be produced on this machine.
- The MSVC compiler `cl` is not on PATH. Visual Studio 2022 Build Tools are present at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`, but that alone is not enough. OpenPose still has to compile Caffe and its Python binding against this interpreter.
- Official Windows binaries are large, tied to older Python versions and specific CUDA builds, and are not something this project can pin as a normal dependency.

A from-source build would mean installing CMake, fighting the Caffe build, and still running on CPU. That is a poor trade for a gesture project that only needs hand landmarks.

## 2026-10-08 — Use RTMLib for RTMPose and RTMW

**Decision:** Install `rtmlib==0.0.16` into `.venv`. Direct dependency is listed in `requirements.txt`.

**Reason:** RTMLib runs the OpenMMLab ONNX exports of RTMPose and RTMW through ONNX Runtime. It does not need CMake, CUDA, or a compiler. It covers both estimators this project needs:

- Dedicated hand estimation: 21 keypoints (`rtmlib.Hand`).
- Whole-body estimation: 133 keypoints, including 21 landmarks on each hand (`rtmlib.Wholebody`, RTMW).

**Resolved packages in `.venv`:**

| Package | Version |
| --- | --- |
| rtmlib | 0.0.16 |
| onnxruntime | 1.30.0 |
| opencv-python | 5.0.0.93 |
| opencv-contrib-python | 5.0.0.93 |
| numpy | 2.4.6 |

## 2026-10-08 — Run on CPU

**Decision:** `backend="onnxruntime"`, `device="cpu"`.

**Reason:** ONNX Runtime on this machine only exposes `CPUExecutionProvider` and `AzureExecutionProvider`. There is no CUDA execution provider because the CUDA toolkit is not installed.

## 2026-10-08 — Download the lightweight checkpoints into the project

**Decision:** Cache weights under `models/hub/checkpoints` by setting `TORCH_HOME` to `D:\gesture-recognition-project\models`. Download the hand pipeline and the lightweight whole-body pipeline. Do not download the balanced or performance RTMW checkpoints.

**Reason:** RTMLib's default cache is outside the project. The lightweight whole-body mode is the one that fits a CPU-only machine. `Hand` only offers a lightweight mode. Larger RTMW modes pull YOLOX-M and the bigger RTMW-DW-X-L network, which are unnecessary until a GPU is available.

**Files downloaded and checked:**

| Role | File | Size | Check |
| --- | --- | --- | --- |
| Hand detector | `rtmdet_nano_8xb32-300e_hand-267f9c8f.onnx` | 3.8 MB | loads |
| Hand pose (RTMPose-m) | `rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx` | 52.5 MB | output shape `(1, 21, 2)` |
| Whole-body detector | `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx` | 19.3 MB | loads |
| Whole-body pose (RTMW) | `rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx` | 123.0 MB | output shape `(1, 133, 2)` |

The shape check ran the pose network on a blank 640×480 image, so it confirms landmark count, not detection quality.

**Default for gesture work:** `Hand`, 21 keypoints. Use `Wholebody(mode="lightweight")` when the body and both hands are needed together.

## 2026-11-12 — Install mpose

**Decision:** Install `mpose==1.2` into `.venv` and add it to `requirements.txt`.

**Reason:** Needed for human action recognition and pose dataset/sequence handling. Installed into `.venv` alongside `rtmlib`, `onnxruntime`, `opencv-python`, and `numpy` without dependency conflicts.

## 2026-11-12 — Download Yoga Poses Dataset

**Decision:** Download `niharika41298/yoga-poses-dataset` into the project directory under `DATASET/` using `kagglehub`. Added `DATASET/` to `.gitignore`.

**Reason:** Provide ground-truth pose classification data (5 classes: downdog, goddess, plank, tree, warrior2). Total 1,551 images (~301.6 MB uncompressed, ~288 MB compressed).
