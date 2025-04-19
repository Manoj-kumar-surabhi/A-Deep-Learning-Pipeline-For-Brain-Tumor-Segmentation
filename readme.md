# 🧠 Multi‑Class Segmentation of Brain Tumor Cells Using U‑Net Models
_A comparison of a manually‑defined TensorFlow‑Keras U‑Net versus a FastAI U‑Net with a pretrained xResNet34 encoder._

---
### 📺 Presentation Video
Watch the full project presentation on YouTube: [(https://youtu.be/Y3WdImIS9yU)](https://youtu.be/Y3WdImIS9yU)


## 🚀 Project Objective
Perform **pixel‑wise classification** of brain‑tumor MRI slices into:

| Label | Description                                     |
|-------|-------------------------------------------------|
| 0     | Background                                      |
| 1     | **NCR/NET** – Necrotic & non‑enhancing core     |
| 2     | **ED** – Peritumoral edema                      |
| 3     | **ET** – Enhancing tumor                        |

Target: **mean Dice ≥ 0.75** for Azure ML Studio deployment.

---

## 📂 Dataset

### BraTS 2020 (Training subset)
- **Download (Kaggle):** [BraTS2020 Dataset](https://www.kaggle.com/datasets/sanglequang/brats2020)
- **Subjects:** 369 training patients  
- **Modalities per patient:** `T1`, `T1ce`, `T2`, `FLAIR`, and `seg`  
- **Volume shape:** 240 × 240 × 155 voxels  
- **Compressed size:** ≈ 3–4 GB (unzipped ≈ 11 GB)

> **Note:** Accept the Kaggle challenge license before downloading.

---

## 🧪 Tools & Frameworks
| Purpose                 | Library                        |
|-------------------------|--------------------------------|
| Deep Learning (TF/Keras)| `tensorflow>=2.8`              |
| Deep Learning (PyTorch) | `torch>=2.0`, `fastai>=2.7`    |
| 3D NIfTI I/O            | `nibabel`                      |
| Image Processing        | `opencv-python`, `Pillow`, `scikit-image` |
| Utilities               | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm` |

---

### Run the notebook
- **FastAI xResNet34 U‑Net:** `u-net_with_xresnet (1).ipynb`


1. Convert 3‑D volumes → 2‑D slices (T1ce + FLAIR channels).
2. Train the respective U‑Net.
3. Report per‑class and mean Dice.

On **Azure ML Studio** upload the notebooks and dataset, attach a GPU compute target (e.g., `Standard_NC6s_v3`), and run cells as‑is.

---

### ⚙️ Training Details

| Parameter      | Manual U‑Net          | FastAI U‑Net               |
| -------------- | --------------------- | -------------------------- |
| **Encoder**    | Conv blocks           | xResNet34 (pretrained)     |
| **Input shape**| 256 × 256 × 2         | 256 × 256 × 2              |
| **Loss**       | Categorical CE + Dice | Categorical CE + Dice      |
| **Optimizer**  | Adam (lr=1e‑4)        | Adam (lr=1e‑3)             |
| **Epochs**     | 25–30                 | 15 (fine‑tune)             |

---

### 📈 Results

| Model                  | Mean Dice | NCR  | ED   | ET   |
| ---------------------- | --------- | ---- | ---- | ---- |
| Manual U‑Net (TF)      | 0.648     | 0.59 | 0.77 | 0.74 |
| xResNet34 U‑Net (FAI)  | 0.747     | 0.65 | 0.81 | 0.78 |

> The pretrained encoder improves small‑region detection (NCR) and lifts the overall mean Dice above the 0.75 deployment threshold.

---

### 💾 requirements.txt

```text
numpy
pandas
scikit-learn
matplotlib
opencv-python
scikit-image
Pillow
nibabel
# TensorFlow stack (manual U‑Net)
tensorflow>=2.8
# PyTorch + FastAI stack (pretrained U‑Net)
torch>=2.0
torchvision
fastai>=2.7
# Utilities
tqdm
