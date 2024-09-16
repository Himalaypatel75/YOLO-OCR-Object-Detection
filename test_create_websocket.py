import cv2
import base64
import asyncio
import websockets
import time

# Settings
video_source = 0  # Webcam or video file path
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

    # Retry connection logic
    retry_attempts = 5
    for attempt in range(retry_attempts):
        try:
            async with websockets.connect(ip_address) as websocket:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    await send_frame(websocket, frame)  # Send the frame over WebSocket
                    time.sleep(interval)
                break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Connection failed: {e}")
            if attempt < retry_attempts - 1:
                print(f"Retrying in 5 seconds... ({attempt + 1}/{retry_attempts})")
                time.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                return

    cap.release()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(video_stream())
