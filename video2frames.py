import cv2
import os

# Input video file path
video_path = r'Path to your video file.mp4'  # Replace with your video file path
# Directory to save frames
output_dir = r'Path to save frames with there respective frames'  # Replace with your desired output directory

# Desired FPS (frames per second)
desired_fps = 2

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the original FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# Calculate the interval between frames
frame_interval = int(fps / desired_fps)

# Frame count initialization
frame_count = 0
saved_frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If no frame is read, the video is finished
    if not ret:
        break

    # Only save the frame if it is the right frame interval
    if frame_count % frame_interval == 0:
        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    # Increment the frame count
    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted at {desired_fps} FPS and saved to {output_dir}")
