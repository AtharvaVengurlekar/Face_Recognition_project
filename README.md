# Face Recognition with ArcFace (InsightFace)

This project performs **face recognition on local videos** using the high-accuracy **ArcFace model** from InsightFace. It detects, embeds, and identifies faces from video frames using precomputed facial embeddings. Ideal for both research and real-world applications.

---

## Key Features

- **Video to Frames**: Automatically extracts frames from input videos.
- **Face Embeddings**: Uses ArcFace via InsightFace to create robust facial embeddings.
- **Image Enhancement**: Improves frame quality using CLAHE, brightness, and contrast adjustments.
- Creating the universal embeddings which can detect in every webcam or cctv footage.
- **Angle Filtering (Optional)**: Filters embeddings based on face orientation for improved accuracy.
- **Face Identification**: Compares faces in test videos with known faces using vector similarity.
- **CPU & GPU Support**: Runs efficiently on both CPU and CUDA-enabled GPUs.

---

## ⚙️ Setup Guide

### Clone the Repository

```bash
git clone https://github.com/<your-username>/Face_Recognition_Project.git
cd Face_Recognition_Project
```

### Create and Activate a Conda Environment

```bash
conda create -n face_recognition_env python=3.10
conda activate face_recognition_env
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

 1. Convert Video to Frames

### Extracts image frames from videos placed in the inputs/ folder.
```bash
python video2frames.py
```

 2. Generate Face Embeddings

Creates ArcFace embeddings for faces detected in the frames.

Embeddings will be stored in either:

- Vector_embeddings_with_angle/ (angle-based Threshold)
- Vector_embeddings_no_angle/ (no Threshold)
```bash
create_face_embeddings_arcface.py
```

 3. Identify Faces in a New Video

Compares new video frames against the known embeddings.
```bash
python face_identification_local_video_arc_face.py
```


