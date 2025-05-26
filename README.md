Absolutely — you want a **clean and concise `README.md`** with:

* Project title
* Key features
* Git clone instructions
* Conda activation
* `pip install`
* Explanation of what the 3 main Python files do
* Commands to run them

Here’s your simplified and professional `README.md`:

---

````markdown
# 🎯 Face Recognition Using ArcFace (InsightFace)

This project performs **face recognition on local videos** using the powerful **ArcFace model** from InsightFace. It processes input videos, detects and embeds faces, and identifies them across frames.

---

## 🚀 Features

- Converts videos into frames.
- Detects and embeds faces using InsightFace (ArcFace).
- Enhances frames (CLAHE, brightness, contrast).
- Optional angle filtering for high-accuracy embeddings.
- Identifies known faces from new videos using precomputed embeddings.
- Works on CPU or GPU (CUDA-enabled).

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Face_Recognition_project.git
cd Face_Recognition_project
````

### 2. Create & Activate Conda Environment

```bash
conda create -n face_recognition_env python=3.10
conda activate face_recognition_env
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📜 Main Scripts & Usage

### 1. `video2frames.py` – Convert Video to Frames

Converts videos from the `inputs/` folder into frames for processing.

```bash
python video2frames.py
```

---

### 2. `create_face_embeddings_arcface.py` – Generate Face Embeddings

Generates facial embeddings (with or without angle filtering) from the extracted frames. Make sure to set `input_dir` and `output_dir` inside the script.

```bash
python create_face_embeddings_arcface.py
```

---

### 3. `face_identification_local_video_arc_face.py` – Face Identification in New Video

Matches detected faces from a test video against the stored embeddings. Customize input and output paths in the script.

```bash
python face_identification_local_video_arc_face.py
```

---



