import cv2
import os
import base64
import asyncio
import websockets
import time
from dotenv import load_dotenv
load_dotenv()

# Settings
video_source = os.getenv('VIDEO_SOURCE') # Webcam or video file pathd
fps = 15  # Frames per second
ip_address = "ws://127.0.0.1:8080"  # WebSocket server address

async def send_frame(websocket, frame):
    _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame in JPG format
    frame_data = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
    await websocket.send(frame_data)

async def video_stream():
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    interval = 1 / fps  # Frame interval based on FPS
    async with websockets.connect(ip_address) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            await send_frame(websocket, frame)  # Send the frame over WebSocket
            time.sleep(interval)

    cap.release()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(video_stream())
