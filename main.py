from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import tensorflow as tf
import os
from deepface import DeepFace

# Load your pre-trained emotion detection model
MODEL = tf.keras.models.load_model('./model.h5')
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion descriptions
EMOTION_DESCRIPTIONS = {
    "Angry": "Displays signs of frustration or irritation",
    "Disgust": "Shows aversion or revulsion",
    "Fear": "Exhibits concern or anxiety",
    "Happy": "Demonstrates joy or contentment",
    "Neutral": "Shows no particular emotional state",
    "Sad": "Displays melancholy or unhappiness",
    "Surprise": "Exhibits astonishment or shock"
}

app = FastAPI()

def process_video(video_path: str, sample_rate: int = 1, detector_backend: str = 'opencv'):
    """
    Processes the video to detect faces and analyze emotions.
    Also extracts the "best face" for attendance verification.
    """
    predictions_data = []
    best_face = None
    best_face_area = 0
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    # Get frame dimensions
    ret, sample_frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Unable to read a frame from the video.")
    frame_height, frame_width = sample_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        try:
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, enforce_detection=False)
        except Exception as e:
            print(f"Error extracting faces: {e}")
            continue

        if faces:
            for face_data in faces:
                facial_area = face_data.get("facial_area", {})
                x = int(facial_area.get("x", 0))
                y = int(facial_area.get("y", 0))
                w = int(facial_area.get("w", 0))
                h = int(facial_area.get("h", 0))
                if w == 0 or h == 0:
                    continue

                # Expand bounding box
                new_w = int(w * 1.2)
                new_h = int(h * 1.8)
                new_x = max(0, x - int((new_w - w) / 2))
                new_y = max(0, y - int(h * 0.2))
                x_end = min(frame_width, new_x + new_w)
                y_end = min(frame_height, new_y + new_h)

                # Crop face region
                face_roi = frame[new_y:y_end, new_x:x_end]
                if face_roi.size == 0:
                    continue

                # Convert to RGB and resize for emotion detection
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (48, 48))
                face_input = face_resized.reshape(1, 48, 48, 3)
                prediction = MODEL.predict(face_input, verbose=0)

                predictions_data.append({
                    'frame': frame_count,
                    'probabilities': prediction[0].tolist()
                })

                # Track best face
                area = new_w * new_h
                if area > best_face_area:
                    best_face_area = area
                    best_face = face_roi.copy()

    cap.release()

    # Save best face (without rotation)
    if best_face is not None:
        cv2.imwrite("best_face.jpg", best_face)

    # Compute average probabilities for emotions
    class_probabilities = {emotion: [] for emotion in CLASS_NAMES}
    for data in predictions_data:
        for idx, prob in enumerate(data['probabilities']):
            class_probabilities[CLASS_NAMES[idx]].append(prob)

    avg_probabilities = {emotion: float(np.mean(class_probabilities[emotion])) if class_probabilities[emotion] else 0 for emotion in CLASS_NAMES}

    # Format results
    results_list = []
    for emotion in EMOTION_DESCRIPTIONS.keys():
        prob = avg_probabilities.get(emotion.lower(), 0)
        results_list.append({
            "name": emotion,
            "value": round(prob, 4),
            "description": EMOTION_DESCRIPTIONS.get(emotion, "")
        })

    return results_list, best_face

def attendance_verification_from_face(best_face, folder_path: str, detector_backend: str = 'opencv'):
    """
    Compares the best detected face against stored images for verification.
    """
    if best_face is None:
        return None

    best_face_path = "best_face.jpg"
    verified_image_name = None
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, image_name)
            try:
                result = DeepFace.verify(img1_path=image_path, img2_path=best_face_path,
                                         model_name="Facenet", enforce_detection=False)
                if result.get("verified"):
                    verified_image_name = image_name
                    break
            except Exception as e:
                print(f"Error verifying {image_name}: {e}")
                continue
    return verified_image_name

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    FastAPI endpoint that:
      - Accepts a video file upload
      - Extracts faces, performs emotion analysis, and detects the best face
      - Verifies the best face against stored images for attendance
    """
    try:
        uploaded_video_path = "uploaded_face_video.mp4"
        contents = await file.read()
        with open(uploaded_video_path, "wb") as f:
            f.write(contents)

        # Process video for emotions & best face
        facial_results, best_face = process_video(uploaded_video_path)

        # Verify attendance using best face
        attendance_image = attendance_verification_from_face(best_face, "../node_backend/uploads")
        attendance_result = {"verified": "Yes", "image": attendance_image} if attendance_image else "No verified image found"

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists("uploaded_face_video.mp4"):
            os.remove("uploaded_face_video.mp4")

    return {
        "Facial Analysis": facial_results,
        "Attendance": attendance_result
    }
