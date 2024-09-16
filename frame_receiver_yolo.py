import cv2
import base64
import numpy as np
import websockets
import asyncio
import os
from ultralytics import YOLO
import time

# YOLOv10 model setup
model = YOLO('yolov10s.pt')  # Load YOLOv10s model

# Output directory for saving images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Video settings
frame_width = 1280  # Set the width of the output video
frame_height = 720  # Set the height of the output video
fps = 15  # Frames per second
output_video_path = "final_output_video.mp4"  # Path to the output video

# Initialize VideoWriter for the final video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# WebSocket Server
async def detect_person(websocket, path):
    frame_counter = 0  # Initialize frame counter
    while True:
        try:
            # Receive base64 frame data
            frame_data = await websocket.recv()
            frame = base64.b64decode(frame_data)
            np_arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode image from base64
            
            # Resize the image to match the video resolution
            img_resized = cv2.resize(img, (frame_width, frame_height))
            
            # Perform person detection with YOLOv10
            results = model.predict(img_resized)  # Run inference on the resized image

            # Save the original image
            output_path_original = os.path.join(output_dir, f"frame_{frame_counter}_original.jpg")
            cv2.imwrite(output_path_original, img_resized)  # Save original resized image
            
            # If you want to only detect people, filter based on class index (0 is 'person' in COCO dataset)
            for result in results:
                boxes = result.boxes  # Extract bounding boxes from result
                for box in boxes:
                    if int(box.cls) == 0:  # Class 0 is 'person'
                        print(f"Person detected: {box}")

            # Plot detected objects on the image
            img_with_detections = results[0].plot()  # Plot the results on the image
            
            # Save the frame with detections to a file
            output_path_detections = os.path.join(output_dir, f"frame_{frame_counter}_detections.jpg")
            cv2.imwrite(output_path_detections, img_with_detections)  # Save image with detections
            
            # Write the frame with detections to the video
            out.write(img_with_detections)

            frame_counter += 1  # Increment frame counter
            
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

# Close the video writer when done
async def main():
    server = await websockets.serve(detect_person, "localhost", 8080)
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(main())
    finally:
        out.release()  # Release the VideoWriter when the server is stopped
        print(f"Video saved at {output_video_path}")
