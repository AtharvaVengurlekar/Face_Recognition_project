# 🎯 Face Recognition with ArcFace (InsightFace)

This project performs **face recognition on local videos** using the high-accuracy **ArcFace model** from InsightFace. It detects, embeds, and identifies faces from video frames using precomputed facial embeddings. Ideal for both research and real-world applications.

---

## ✨ Key Features

- 🎥 **Video to Frames**: Automatically extracts frames from input videos.
- 🧠 **Face Embeddings**: Uses ArcFace via InsightFace to create robust facial embeddings.
- 🎛️ **Image Enhancement**: Improves frame quality using CLAHE, brightness, and contrast adjustments.
- 🎯 **Angle Filtering (Optional)**: Filters embeddings based on face orientation for improved accuracy.
- 🧍‍♂️ **Face Identification**: Compares faces in test videos with known faces using vector similarity.
- ⚡ **CPU & GPU Support**: Runs efficiently on both CPU and CUDA-enabled GPUs.

---

## ⚙️ Setup Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Face_Recognition_Project.git
cd Face_Recognition_Project



