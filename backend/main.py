import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.inference import predict
from backend.tts import speak

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_keypoints(result):
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y] for lm in hand]).flatten()
    return np.zeros(42)

@app.get("/")
def root():
    return {"status": "ISL backend running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence = []
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            if len(sequence) == 30:
                word, confidence = predict(np.array(sequence))
                sequence = []
                audio = speak(word)
                response = {
                    "word": word,
                    "confidence": confidence,
                    "has_audio": audio is not None
                }
                await websocket.send_json(response)
                if audio:
                    await websocket.send_bytes(audio)

    except Exception as e:
        print(f"Connection closed: {e}")