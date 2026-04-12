import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']
SEQUENCES_PER_SIGN = 30
FRAMES_PER_SEQUENCE = 30
DATA_PATH = os.path.join('data', 'sequences')

for sign in SIGNS:
    os.makedirs(os.path.join(DATA_PATH, sign), exist_ok=True)

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

cap = cv2.VideoCapture(0)

for sign in SIGNS:
    existing = len(os.listdir(os.path.join(DATA_PATH, sign)))
    if existing >= SEQUENCES_PER_SIGN:
        print(f"Skipping {sign} — already have {existing} sequences")
        continue

    for seq in range(existing, SEQUENCES_PER_SIGN):

        # Wait for spacebar
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"NEXT: {sign.upper()}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.putText(frame, "Press SPACE when ready", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Seq {seq+1}/{SEQUENCES_PER_SIGN}", (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.imshow("ISL Data Collection", frame)
            key = cv2.waitKey(1)
            if key == 32:  # spacebar
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Quit early.")
                exit()

        # Record 30 frames
        sequence = []
        for frame_num in range(FRAMES_PER_SEQUENCE):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for lm in hand:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cv2.putText(frame, f"RECORDING: {sign.upper()}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Frame {frame_num+1}/{FRAMES_PER_SEQUENCE}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("ISL Data Collection", frame)
            cv2.waitKey(1)

        np.save(os.path.join(DATA_PATH, sign, f"{seq}.npy"), np.array(sequence))
        print(f"Saved {sign} — sequence {seq+1}/{SEQUENCES_PER_SIGN}")

cap.release()
cv2.destroyAllWindows()
print("\nAll done!")