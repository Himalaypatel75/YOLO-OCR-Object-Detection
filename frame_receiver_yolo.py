import cv2
import base64
import numpy as np
import websockets
import asyncio
import os
from ultralytics import YOLO
import urllib.parse

# YOLOv10 model setup
model = YOLO('yolov10s.pt')  # Load YOLOv10s model

# Output directory for saving images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Video settings
frame_width = 1280  # Set the width of the output video
frame_height = 720  # Set the height of the output video
fps = 5  # Frames per second

# Dictionary to store VideoWriter for each camera ID
video_writers = {}

# WebSocket Server
async def detect_person(websocket, path):
    # Parse query parameters
    query = urllib.parse.urlparse(path).query
    camera_id = urllib.parse.parse_qs(query).get('camera_id', [None])[0]

    print(f"Connected to camera ID: {camera_id}")  # Print or log the camera ID

    # Initialize VideoWriter for this camera ID if it doesn't exist
    if camera_id not in video_writers:
        output_video_path = f"{camera_id}_output_video.mp4"  # Create a unique output path for each camera
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        video_writers[camera_id] = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0  # Initialize frame counter
    while True:
        try:
            # if frame_counter > 600:  # Limit to 600 frames
            #     break
            
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
            output_path_original = os.path.join(output_dir, f"frame_{frame_counter}_{camera_id}_original.jpg")
            cv2.imwrite(output_path_original, img_resized)  # Save original resized image
            
            # Detect persons and log them
            for result in results:
                boxes = result.boxes  # Extract bounding boxes from result
                for box in boxes:
                    if int(box.cls) == 0:  # Class 0 is 'person'
                        print(f"Person detected in camera ID {camera_id}: {box}")

            # Plot detected objects on the image
            img_with_detections = results[0].plot()  # Plot the results on the image
            
            # Save the frame with detections to a file
            output_path_detections = os.path.join(output_dir, f"frame_{frame_counter}_{camera_id}_detections.jpg")
            cv2.imwrite(output_path_detections, img_with_detections)  # Save image with detections
            
            # Write the frame with detections to the video for this camera ID
            video_writers[camera_id].write(img_with_detections)

            frame_counter += 1  # Increment frame counter
            
        except websockets.ConnectionClosed:
            print(f"Connection closed for camera ID: {camera_id}")
            break

# Close the video writers when done
async def main():
    server1 = await websockets.serve(detect_person, "localhost", 8080)
    server2 = await websockets.serve(detect_person, "localhost", 8081)  # Different port for another camera
    await asyncio.gather(server1.wait_closed(), server2.wait_closed())

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(main())
    finally:
        # Release all VideoWriters
        for vw in video_writers.values():
            vw.release()
        print("All video files saved.")
