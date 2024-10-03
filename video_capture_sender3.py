import cv2
import os
import base64
import asyncio
import websockets
import time
from dotenv import load_dotenv

load_dotenv()

# Settings
video_source = os.getenv('VIDEO_SOURCE_2')  # Webcam or video file path
fps = 5  # Frames per second
camera_id = 4
ip_address = f"ws://127.0.0.1:8080?camera_id={camera_id}"   # WebSocket server address

async def send_frame(websocket, frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame in JPG format
        frame_data = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
        await websocket.send(frame_data)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
        raise  # Re-raise the exception to handle it in the main loop
    except Exception as e:
        print(f"An error occurred while sending frame: {e}")

async def video_stream():
    while True:  # Loop to reconnect on failure
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print("Error: Cannot open video source.")
                return

            interval = 1 / fps  # Frame interval based on FPS
            last_sent_time = time.time()  # Track the last time a frame was sent

            async with websockets.connect(ip_address) as websocket:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to read frame from video source.")
                        break

                    current_time = time.time()
                    if current_time - last_sent_time >= interval:  # Check if the interval has passed
                        await send_frame(websocket, frame)  # Send the frame over WebSocket
                        last_sent_time = current_time  # Update the last sent time

            cap.release()

        except (cv2.error, websockets.exceptions.ConnectionClosed) as e:
            print(f"Error occurred: {e}. Attempting to reconnect...")
            await asyncio.sleep(2)  # Wait before reconnecting
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await asyncio.sleep(2)  # Wait before reconnecting

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(video_stream())
