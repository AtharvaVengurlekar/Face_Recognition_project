# full detection and recognition pipeline 
import cv2
import numpy as np
import os
import cupy as cp
from insightface.app import FaceAnalysis

# Load saved embeddings directly to GPU using CuPy
def load_saved_embeddings(embedding_dir):
    embeddings = []
    names = []
    
    for person in os.listdir(embedding_dir):
        person_dir = os.path.join(embedding_dir, person)
        if os.path.isdir(person_dir):
            for file in os.listdir(person_dir):
                if file.endswith(".npy"):
                    path = os.path.join(person_dir, file)
                    # Load embeddings directly into GPU memory using CuPy
                    embedding = cp.load(path)  # Use cp.load instead of np.load
                    embeddings.append(embedding)
                    names.append(person)
    
    # Stack all embeddings into a single CuPy array on GPU
    embeddings_gpu = cp.stack(embeddings)
    return embeddings_gpu, names

# Calculate Cosine similarity between all embeddings at once using CuPy (GPU)
def batch_cosine_similarity(embeddings1, embeddings2):
    embeddings1_gpu = cp.asarray(embeddings1)
    embeddings2_gpu = cp.asarray(embeddings2)
    
    # Ensure both embeddings are 2D arrays
    if embeddings1_gpu.ndim == 1:
        embeddings1_gpu = cp.expand_dims(embeddings1_gpu, axis=0)  # Convert to 2D
    if embeddings2_gpu.ndim == 1:
        embeddings2_gpu = cp.expand_dims(embeddings2_gpu, axis=0)  # Convert to 2D
    
    # Compute dot product between all pairs of embeddings
    dot_product = cp.matmul(embeddings1_gpu, embeddings2_gpu.T)
    
    # Compute norms of embeddings
    norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
    norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)
    
    # Compute cosine similarity
    cosine_similarity_matrix = dot_product / (norm1 * norm2.T)
    
    return cosine_similarity_matrix

# Find the closest match in the batch for all embeddings
#def find_closest_match_batch(embedding, saved_embeddings, names, threshold=0.5):
def find_closest_match_batch(embedding, saved_embeddings, names, threshold=0.4):    
    # Ensure embedding is a 2D array
    embedding_reshaped = np.expand_dims(embedding, axis=0)  # Convert to 2D
    
    # Compute cosine similarity matrix between the current embedding and all saved embeddings
    cosine_similarity_matrix = batch_cosine_similarity(embedding_reshaped, saved_embeddings)
    
    # Get the highest similarity and its index
    max_similarity = cp.max(cosine_similarity_matrix, axis=1)
    closest_index = cp.argmax(cosine_similarity_matrix, axis=1)
    
    # Extract the closest matching name and similarity score
    closest_index = closest_index.item()  # Convert CuPy ndarray to Python scalar index
    closest_match = names[closest_index]  # Use the scalar index to retrieve the name
    similarity = max_similarity.item()  # Convert CuPy ndarray to scalar value
    
    # Check if similarity exceeds threshold
    if similarity >= threshold:
        label = f"{closest_match} ({similarity:.2f})"
    else:
        label = "Unknown"
    
    return label, similarity

# Enhance the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

# Optional: More aggressive contrast & brightness control
def adjust_brightness_contrast(image, alpha=1.2, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# Deblur the image using Unsharp Mask technique
def deblur_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return unsharp_mask

# Process video and match faces with saved embeddings
def process_video_and_search(input_video_path, output_video_path, embedding_dir):
    # Load saved embeddings into memory and transfer them to GPU
    print("Loading embeddings into memory...")
    embeddings_gpu, names = load_saved_embeddings(embedding_dir)
    print(f"Loaded {len(names)} embeddings into GPU memory.")

    # Initialize the InsightFace app for face analysis (ArcFace)
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))  # Load model on GPU
    
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print("Error: Couldn't open video file.")
        return

    # Get video properties (for saving the output video)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for compatibility with MP4 format
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Apply image preprocessing (deblur and enhance)
        deblurred_frame = deblur_image(frame)
        enhanced_frame = enhance_image(deblurred_frame)

        # Extract face embeddings using ArcFace
        faces = app.get(enhanced_frame)

        for face in faces:
            face_embedding = np.array(face.embedding)  # ArcFace embedding
            if face_embedding is None:
                continue

            # Find the closest match for the current face using batch cosine similarity
            closest_match, similarity = find_closest_match_batch(face_embedding, embeddings_gpu, names)
            
            # Label the face based on similarity
            xmin, ymin, xmax, ymax = face.bbox
            print(f"Face bounding box: {xmin}, {ymin}, {xmax}, {ymax}")
            
            # Draw a box around the face on processed frame
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Label the face on processed frame
            cv2.putText(frame, closest_match, (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out_video.write(frame)

    # Release the video capture and writer objects
    video_capture.release()
    out_video.release()

    # Close any OpenCV windows
    cv2.destroyAllWindows()

    print("Video processing complete. Output saved to:", output_video_path)

# Example usage
input_video_path = r"Path to the input video where you want to detec and recognize"  # Replace with your input video path
output_video_path = r"Path where you want to save the output video "  # Replace with your desired output video path
embedding_dir = r"Path where you embeddinsg are created"   # Replace with your embeddings folder

process_video_and_search(input_video_path, output_video_path, embedding_dir)













# # angle threshold for front facing detection
# import os
# import cv2
# import numpy as np
# import time
# from insightface.app import FaceAnalysis

# try:
#     import cupy as cp
#     HAS_CUDA = True
#     print("CUDA is available. Using GPU acceleration.")
# except ImportError:
#     cp = np
#     HAS_CUDA = False
#     print("CUDA is not available. Using CPU fallback.")

# def load_saved_embeddings(embedding_dir):
#     print(f"Loading embeddings from {embedding_dir}...")
#     if not os.path.exists(embedding_dir):
#         raise FileNotFoundError(f"Embedding directory '{embedding_dir}' does not exist.")
#     embeddings = []
#     names = []
#     for person in os.listdir(embedding_dir):
#         person_dir = os.path.join(embedding_dir, person)
#         if not os.path.isdir(person_dir):
#             continue
#         for file in os.listdir(person_dir):
#             if file.endswith(".npy"):
#                 path = os.path.join(person_dir, file)
#                 embedding_np = np.load(path)
#                 embeddings.append(cp.asarray(embedding_np) if HAS_CUDA else embedding_np)
#                 names.append(person)
#     if not embeddings:
#         raise ValueError("No valid embeddings found.")
#     embeddings_array = cp.stack(embeddings) if HAS_CUDA else np.stack(embeddings)
#     return embeddings_array, names

# def batch_cosine_similarity(embedding, embeddings_array):
#     if HAS_CUDA:
#         embedding = cp.asarray(embedding)
#         if embedding.ndim == 1:
#             embedding = cp.expand_dims(embedding, axis=0)
#         dot = cp.matmul(embedding, embeddings_array.T)
#         norm1 = cp.linalg.norm(embedding, axis=1, keepdims=True)
#         norm2 = cp.linalg.norm(embeddings_array, axis=1, keepdims=True)
#         return dot / (norm1 * norm2.T)
#     else:
#         if embedding.ndim == 1:
#             embedding = np.expand_dims(embedding, axis=0)
#         dot = np.matmul(embedding, embeddings_array.T)
#         norm1 = np.linalg.norm(embedding, axis=1, keepdims=True)
#         norm2 = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
#         return dot / (norm1 * norm2.T)

# # def find_closest_match(embedding, saved_embeddings, names, threshold=0.4):
# def find_closest_match(embedding, saved_embeddings, names, threshold=0.35):
#     sim = batch_cosine_similarity(embedding, saved_embeddings)
#     if HAS_CUDA:
#         max_sim = cp.max(sim).item()
#         index = cp.argmax(sim).item()
#     else:
#         max_sim = np.max(sim)
#         index = np.argmax(sim)
    
#     if max_sim >= threshold:
#         # Return matched name with similarity
#         return (names[index], max_sim)
#     else:
#         # similarity too low - no recognition
#         return (None, max_sim)
 
# # def is_front_facing(pose, yaw_threshold=30, pitch_threshold=30, roll_threshold=30):
# #     pitch, yaw, roll = pose  # Check pose order carefully
# #     return all(abs(angle) <= threshold for angle, threshold in zip((yaw, pitch, roll), (yaw_threshold, pitch_threshold, roll_threshold)))

# #approved my rutwik, 
# def is_front_facing(pose, yaw_threshold=25, pitch_threshold=25, roll_threshold=25):
#     pitch, yaw, roll = pose
#     return all(abs(angle) <= threshold for angle, threshold in zip((yaw, pitch, roll), (yaw_threshold, pitch_threshold, roll_threshold)))

# def enhance_image(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# def adjust_brightness_contrast(image, alpha=1.05, beta=3):
#     return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# def process_video_and_search(input_video_path, output_video_path, embedding_dir, threshold=0.4):
#     if not os.path.exists(input_video_path):
#         raise FileNotFoundError(f"Video file '{input_video_path}' not found.")

#     embeddings_array, names = load_saved_embeddings(embedding_dir)

#     app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     try:
#         app.prepare(ctx_id=0, det_size=(640, 640))
#         print("Initialized FaceAnalysis with GPU.")
#     except Exception as e:
#         print("GPU failed, switching to CPU.")
#         app.prepare(ctx_id=-1, det_size=(640, 640))

#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         raise IOError(f"Failed to open video '{input_video_path}'.")

#     width, height = int(cap.get(3)), int(cap.get(4))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     print(f"Processing video: {input_video_path} ({total} frames)")
#     frame_count, detected, recognized, unknown = 0, 0, 0, 0
#     start = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         enhanced = enhance_image(frame)
#         enhanced = adjust_brightness_contrast(enhanced)

#         faces = app.get(enhanced)

#         # #approved by rutwik
#         # for face in faces:
#         #     if not is_front_facing(face.pose):
#         #         continue
#         #     embedding = np.array(face.embedding)
#         #     name, sim = find_closest_match(embedding, embeddings_array, names, threshold)
#         #     if sim < threshold:
#         #         # similarity too low - ignore this detection (do not draw box/label)
#         #         continue
            
#         #     # Only draw if similarity >= threshold
#         #     xmin, ymin, xmax, ymax = face.bbox.astype(int)
#         #     color = (0, 255, 0)  # green for known face
#         #     text = f"{name} ({sim:.2f})"
            
#         #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#         #     cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#         # out.write(frame)

#         for face in faces:
#             if not is_front_facing(face.pose):
#                 continue

#             embedding = np.array(face.embedding)
#             name, sim = find_closest_match(embedding, embeddings_array, names, threshold)
#             xmin, ymin, xmax, ymax = face.bbox.astype(int)

#             if name is None or sim < threshold:
#                 color = (0, 255, 0)  # Red for unknown
#                 text = "Unknown"
#                 unknown += 1
#             else:
#                 color = (0, 255, 0)  # Green for recognized
#                 text = f"{name} ({sim:.2f})"
#                 recognized += 1

#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#             cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         out.write(frame)


#         if frame_count % 100 == 0:
#             elapsed = time.time() - start
#             print(f"Processed {frame_count}/{total} frames at {frame_count / elapsed:.2f} FPS")

#     cap.release()
#     out.release()
#     print("Video processing complete.")
#     print(f"Frames: {frame_count}, Detected: {detected}, Recognized: {recognized}, Unknown: {unknown}")



# def extract_frames(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     os.makedirs(output_dir, exist_ok=True)

#     count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_path = os.path.join(output_dir, f"{count:06d}.jpg")
#         cv2.imwrite(frame_path, frame)
#         count += 1

#         if count % 10 == 0 or count == total_frames:
#             print(f"Extracted {count}/{total_frames} frames")

#     cap.release()
#     print("✅ Done extracting frames.")    

# if __name__ == "__main__":
#     # Example usage — update with your actual paths
#     input_video_path = r"Path to the input video where you want to detec and recognize"  # Replace with your input video path
#     output_video_path = r"Path where you want to save the output video "  # Replace with your desired output video path
#     embedding_dir = r"Path where you embeddinsg are created"   # Replace with your embeddings folder
#     process_video_and_search(input_video, output_video, embeddings_dir)




