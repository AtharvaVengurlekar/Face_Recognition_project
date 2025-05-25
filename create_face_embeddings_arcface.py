#embeddings with 30 degree threshold
#claud ai 
# import os
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis

# def load_insightface_model():
#     """Load and prepare the InsightFace model with proper error handling"""
#     app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     try:
#         # Try to use GPU first
#         app.prepare(ctx_id=0, det_size=(640, 640))
#         print("Using GPU for face analysis")
#     except Exception as e:
#         # Fall back to CPU if GPU fails
#         print(f"GPU initialization failed: {e}. Falling back to CPU.")
#         app.prepare(ctx_id=-1, det_size=(640, 640))
#         print("Using CPU for face analysis")
#     return app

# def upscale_image_if_needed(face_img, min_size=112):
#     """Upscale the image if it's smaller than the minimum required size"""
#     if face_img is None or face_img.size == 0:
#         return None
    
#     h, w = face_img.shape[:2]
#     if min(h, w) < min_size:
#         scale = min_size / min(h, w)
#         new_w, new_h = int(w * scale), int(h * scale)
#         return cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#     return face_img

# def enhance_image(image):
#     """Enhance image using CLAHE"""
#     if image is None:
#         return None
    
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     merged_lab = cv2.merge((cl, a, b))
#     enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
#     return enhanced_bgr

# def adjust_brightness_contrast(image, alpha=1.1, beta=5):
#     """Adjust brightness and contrast of the image"""
#     if image is None:
#         return None
    
#     return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# def is_front_facing(pose, yaw_threshold=30, pitch_threshold=30, roll_threshold=30):
#     """Check if face is within ±30 degrees for all angles"""
#     yaw_angle = pose[1]   # Left-right
#     roll_angle = pose[2]  # Tilt
#     pitch_angle = pose[0] # Up-down
    
#     return (abs(yaw_angle) <= yaw_threshold and 
#             abs(pitch_angle) <= pitch_threshold and 
#             abs(roll_angle) <= roll_threshold)

# def create_face_embeddings(input_dir, output_dir):
#     """Create face embeddings for all images in input directory"""
#     if not os.path.exists(input_dir):
#         raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
#     os.makedirs(output_dir, exist_ok=True)
#     app = load_insightface_model()
    
#     # Track statistics
#     total_images = 0
#     processed_images = 0
#     skipped_images = 0
#     face_count = 0

#     for person in os.listdir(input_dir):
#         person_folder = os.path.join(input_dir, person)
#         if os.path.isdir(person_folder):
#             print(f"Processing folder: {person}")
#             output_person_dir = os.path.join(output_dir, person)
#             os.makedirs(output_person_dir, exist_ok=True)
            
#             person_images = [f for f in os.listdir(person_folder) 
#                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
#             total_images += len(person_images)
            
#             for img_file in person_images:
#                 img_path = os.path.join(person_folder, img_file)
#                 try:
#                     frame = cv2.imread(img_path)
#                     if frame is None:
#                         print(f"Could not read image: {img_file}")
#                         skipped_images += 1
#                         continue
                    
#                     # Apply image enhancements for better face detection
#                     frame = enhance_image(frame)
#                     frame = adjust_brightness_contrast(frame)  
#                     frame = upscale_image_if_needed(frame)
                    
#                     faces = app.get(frame)
#                     if not faces:
#                         print(f"No face found in {img_file}")
#                         skipped_images += 1
#                         continue
                    
#                     person_face_count = 0
                    
#                     for idx, face in enumerate(faces):
#                         # Check if face is front-facing within the threshold
#                         if not is_front_facing(face.pose):
#                             continue
                        
#                         # Get the face embedding and normalize it
#                         embedding = np.array(face.embedding)
#                         embedding = embedding / np.linalg.norm(embedding)  # Normalize
                        
#                         # Save the embedding
#                         save_path = os.path.join(output_person_dir, 
#                                                 f"{os.path.splitext(img_file)[0]}_face{idx}_embedding.npy")
#                         np.save(save_path, embedding)
                        
#                         person_face_count += 1
#                         face_count += 1
                    
#                     if person_face_count > 0:
#                         processed_images += 1
#                         print(f"Saved {person_face_count} embeddings from {img_file}")
#                     else:
#                         print(f"No suitable faces found in {img_file} (pose threshold not met)")
#                         skipped_images += 1
                
#                 except Exception as e:
#                     print(f"Error processing {img_file}: {e}")
#                     skipped_images += 1
    
#     print("\nSummary:")
#     print(f"Total images found: {total_images}")
#     print(f"Successfully processed images: {processed_images}")
#     print(f"Skipped images: {skipped_images}")
#     print(f"Total face embeddings created: {face_count}")

# if __name__ == "__main__":
#     input_dir = r"Path to Folder where teh frames are stored"
#     output_dir = r"Path to where you want to save the embeddings"
#     create_face_embeddings(input_dir, output_dir)






# #no angle embeddings 
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_insightface_model():
    """Load and prepare the InsightFace model with proper error handling"""
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Using GPU for face analysis")
    except Exception as e:
        print(f"GPU initialization failed: {e}. Falling back to CPU.")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("Using CPU for face analysis")
    return app

def upscale_image_if_needed(face_img, min_size=112):
    if face_img is None or face_img.size == 0:
        return None
    h, w = face_img.shape[:2]
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return face_img

def enhance_image(image):
    if image is None:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def adjust_brightness_contrast(image, alpha=1.1, beta=5):
    if image is None:
        return None
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def create_face_embeddings(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    os.makedirs(output_dir, exist_ok=True)
    app = load_insightface_model()

    total_images = 0
    processed_images = 0
    skipped_images = 0
    face_count = 0

    for person in os.listdir(input_dir):
        person_folder = os.path.join(input_dir, person)
        if os.path.isdir(person_folder):
            print(f"Processing folder: {person}")
            output_person_dir = os.path.join(output_dir, person)
            os.makedirs(output_person_dir, exist_ok=True)

            person_images = [f for f in os.listdir(person_folder)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            total_images += len(person_images)

            for img_file in person_images:
                img_path = os.path.join(person_folder, img_file)
                try:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        print(f"Could not read image: {img_file}")
                        skipped_images += 1
                        continue

                    frame = enhance_image(frame)
                    frame = adjust_brightness_contrast(frame)
                    frame = upscale_image_if_needed(frame)

                    faces = app.get(frame)
                    if not faces:
                        print(f"No face found in {img_file}")
                        skipped_images += 1
                        continue

                    for idx, face in enumerate(faces):
                        embedding = np.array(face.embedding)
                        embedding = embedding / np.linalg.norm(embedding)
                        save_path = os.path.join(
                            output_person_dir,
                            f"{os.path.splitext(img_file)[0]}_face{idx}_embedding.npy"
                        )
                        np.save(save_path, embedding)
                        face_count += 1

                    processed_images += 1
                    print(f"Saved {len(faces)} embeddings from {img_file}")

                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    skipped_images += 1

    print("\nSummary:")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed images: {processed_images}")
    print(f"Skipped images: {skipped_images}")
    print(f"Total face embeddings created: {face_count}")

if __name__ == "__main__":
    input_dir = r"Path to Folder where teh frames are stored"
    output_dir = r"Path to where you want to save the embeddings"
    create_face_embeddings(input_dir, output_dir)
