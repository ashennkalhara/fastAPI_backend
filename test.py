from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import tensorflow as tf
import os
from deepface import DeepFace
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load your pre-trained emotion detection model with error handling
try:
    MODEL = tf.keras.models.load_model('./model.h5')
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise e

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

# Create a static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Save the HTML UI to a file in the static directory
with open("static/index.html", "w") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Analysis System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/alpinejs/3.10.3/cdn.min.js" defer></script>
    <style>
        .bg-theme {
            background-color: #3498db;
        }
        .bg-theme-light {
            background-color: #e1f0fa;
        }
        .text-theme {
            color: #3498db;
        }
        .border-theme {
            border-color: #3498db;
        }
        .hover\:bg-theme-dark:hover {
            background-color: #2980b9;
        }
        .hover\:border-theme:hover {
            border-color: #3498db;
        }
        .hover\:bg-theme-light:hover {
            background-color: #e1f0fa;
        }
        .focus\:ring-theme:focus {
            --tw-ring-color: #3498db;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div x-data="videoAnalysisApp()" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <h1 class="text-3xl font-bold text-theme">Facial Analysis System</h1>
                <p class="text-gray-600 mt-2 md:mt-0">Emotion Detection & Attendance Verification</p>
            </div>
            <div class="h-1 w-full bg-theme mt-4 rounded-full"></div>
        </header>

        <!-- Main Content -->
        <main class="mb-10">
            <div class="bg-white shadow-xl rounded-xl overflow-hidden border border-gray-200">
                <!-- Upload Section -->
                <div class="p-8 border-b border-gray-200">
                    <h2 class="text-2xl font-semibold mb-6 text-gray-800 flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2 text-theme" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        Upload Video for Analysis
                    </h2>
                    <div class="space-y-6">
                        <div 
                            class="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center cursor-pointer transition-all duration-300 transform hover:scale-[1.01] hover:shadow-md hover:bg-theme-light hover:border-theme"
                            :class="{'border-theme bg-theme-light': isDragging}"
                            @dragover.prevent="isDragging = true"
                            @dragleave.prevent="isDragging = false"
                            @drop.prevent="handleFileDrop"
                            @click="document.getElementById('videoInput').click()"
                        >
                            <div x-show="!selectedFile && !isProcessing">
                                <svg class="mx-auto h-16 w-16 text-theme opacity-80" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                                </svg>
                                <p class="mt-4 text-base text-gray-600">
                                    Drag and drop a video file or <span class="text-theme font-medium">click to browse</span>
                                </p>
                                <p class="mt-2 text-sm text-gray-400">MP4, MOV, or AVI format recommended</p>
                            </div>
                            <div x-show="selectedFile && !isProcessing" class="flex flex-col items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                                </svg>
                                <p class="mt-4 text-lg font-medium text-gray-900" x-text="selectedFile.name"></p>
                                <p class="text-sm text-gray-500 mt-1" x-text="formatFileSize(selectedFile.size)"></p>
                                <button @click.stop="selectedFile = null" class="mt-4 text-sm text-red-500 hover:text-red-700 flex items-center">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                    Remove
                                </button>
                            </div>
                            <div x-show="isProcessing" class="flex flex-col items-center">
                                <svg class="animate-spin h-16 w-16 text-theme" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <p class="mt-4 text-lg font-medium text-gray-900">Processing your video...</p>
                                <p class="text-sm text-gray-500 mt-1">This may take a moment</p>
                                <div class="mt-4 w-48 h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <div class="h-full bg-theme animate-pulse"></div>
                                </div>
                            </div>
                        </div>
                        <input type="file" id="videoInput" class="hidden" accept="video/*" @change="handleFileSelect">
                        <button 
                            @click="uploadVideo" 
                            class="w-full bg-theme hover:bg-theme-dark text-white font-medium py-3 px-6 rounded-lg transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-theme focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg text-lg"
                            :disabled="!selectedFile || isProcessing"
                        >
                            <span class="flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                                Analyze Video
                            </span>
                        </button>
                    </div>
                </div>

                <!-- Results Section -->
                <div x-show="analysisResults" class="p-8 bg-theme-light">
                    <h2 class="text-2xl font-semibold mb-6 text-gray-800 flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2 text-theme" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Analysis Results
                    </h2>
                    
                    <!-- Attendance Section -->
                    <div class="mb-8 p-6 bg-white rounded-xl shadow-md border border-gray-100">
                        <h3 class="text-xl font-medium text-gray-800 mb-4 flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-theme" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            Attendance Verification
                        </h3>
                        <div x-show="analysisResults && analysisResults.Attendance && analysisResults.Attendance.verified === 'Yes'" class="p-4 bg-green-50 rounded-lg border border-green-100">
                            <div class="flex items-center text-green-600">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span class="font-semibold text-lg">Attendance Verified</span>
                            </div>
                            <p class="mt-2 text-base text-gray-600">
                                Matched with: <span class="font-medium" x-text="analysisResults.Attendance.image"></span>
                            </p>
                        </div>
                        <div x-show="analysisResults && (!analysisResults.Attendance || analysisResults.Attendance === 'No verified image found')" class="p-4 bg-red-50 rounded-lg border border-red-100">
                            <div class="flex items-center text-red-600">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span class="font-semibold text-lg">No Match Found</span>
                            </div>
                            <p class="mt-2 text-base text-gray-600">
                                Unable to verify attendance with the provided video.
                            </p>
                        </div>
                    </div>
                    
                    <!-- Emotion Analysis Section -->
                    <div class="p-6 bg-white rounded-xl shadow-md border border-gray-100">
                        <h3 class="text-xl font-medium text-gray-800 mb-4 flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2 text-theme" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Facial Emotion Analysis
                        </h3>
                        <div class="space-y-6">
                            <template x-for="(emotion, index) in analysisResults?.['Facial Analysis']" :key="index">
                                <div class="space-y-2 p-3 rounded-lg hover:bg-gray-50 transition-colors duration-200">
                                    <div class="flex justify-between items-center">
                                        <span class="text-base font-medium text-gray-700" x-text="emotion.name"></span>
                                        <span class="text-base font-medium text-theme" x-text="formatPercentage(emotion.value)"></span>
                                    </div>
                                    <div class="w-full bg-gray-200 rounded-full h-3">
                                        <div class="bg-theme h-3 rounded-full transition-all duration-500 ease-out" :style="`width: ${emotion.value * 100}%`"></div>
                                    </div>
                                    <p class="text-sm text-gray-600 mt-1" x-text="emotion.description"></p>
                                </div>
                            </template>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer class="text-center mt-12">
            <div class="border-t border-gray-200 pt-8 pb-6">
                <p class="text-gray-600">&copy; 2025 Facial Analysis System. All rights reserved.</p>
                <div class="flex justify-center mt-4 space-x-4">
                    <a href="#" class="text-theme hover:text-theme-dark">Privacy Policy</a>
                    <a href="#" class="text-theme hover:text-theme-dark">Terms of Service</a>
                    <a href="#" class="text-theme hover:text-theme-dark">Support</a>
                </div>
            </div>
        </footer>
    </div>

    <script>
        function videoAnalysisApp() {
            return {
                isDragging: false,
                selectedFile: null,
                isProcessing: false,
                analysisResults: null,
                
                handleFileDrop(e) {
                    this.isDragging = false;
                    if (e.dataTransfer.files.length) {
                        const file = e.dataTransfer.files[0];
                        if (file.type.startsWith('video/')) {
                            this.selectedFile = file;
                        } else {
                            alert('Please upload a video file.');
                        }
                    }
                },
                
                handleFileSelect(e) {
                    if (e.target.files.length) {
                        this.selectedFile = e.target.files[0];
                    }
                },
                
                formatFileSize(bytes) {
                    if (bytes === 0) return '0 Bytes';
                    const k = 1024;
                    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
                },
                
                formatPercentage(value) {
                    return (value * 100).toFixed(1) + '%';
                },
                
                uploadVideo() {
                    if (!this.selectedFile) return;
                    
                    this.isProcessing = true;
                    this.analysisResults = null;
                    
                    const formData = new FormData();
                    formData.append('file', this.selectedFile);
                    
                    fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        this.analysisResults = data;
                        console.log('Analysis results:', data);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred during analysis. Please try again.');
                    })
                    .finally(() => {
                        this.isProcessing = false;
                    });
                }
            };
        }
    </script>
</body>
</html>""")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# MongoDB configuration
MONGO_URI = "mongodb+srv://ashennkalhara:ashenn2178@studentdetails.6i8wy.mongodb.net/"
DB_NAME = "test"
COLLECTION_NAME = "students"

def process_video(video_path: str, sample_rate: int = 1, detector_backend: str = 'opencv'):
    """
    Processes the video to detect faces and analyze emotions.
    Also extracts the "best face" for attendance verification.
    """
    predictions_data = []
    best_face = None
    best_face_area = 0
    frame_count = 0

    # Initialize video capture with error handling
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
    except Exception as e:
        logging.error(f"Error initializing video capture: {e}")
        raise e

    try:
        ret, sample_frame = cap.read()
        if not ret:
            raise Exception("Unable to read a frame from the video.")
        frame_height, frame_width = sample_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception as e:
        cap.release()
        logging.error(f"Error processing first frame: {e}")
        raise e

    # Process video frames
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
        except Exception as e:
            logging.error(f"Error reading frame {frame_count}: {e}")
            break

        try:
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, enforce_detection=False)
        except Exception as e:
            logging.error(f"Error extracting faces from frame {frame_count}: {e}")
            continue

        if faces:
            for face_data in faces:
                try:
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

                    # Predict emotion
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
                except Exception as e:
                    logging.error(f"Error processing face in frame {frame_count}: {e}")
                    continue

    cap.release()

    # Save best face (if available)
    try:
        if best_face is not None:
            cv2.imwrite("best_face.jpg", best_face)
    except Exception as e:
        logging.error(f"Error saving best face image: {e}")

    # Compute average probabilities for emotions
    try:
        class_probabilities = {emotion: [] for emotion in CLASS_NAMES}
        for data in predictions_data:
            for idx, prob in enumerate(data['probabilities']):
                class_probabilities[CLASS_NAMES[idx]].append(prob)
        avg_probabilities = {
            emotion: float(np.mean(class_probabilities[emotion])) if class_probabilities[emotion] else 0 
            for emotion in CLASS_NAMES
        }
    except Exception as e:
        logging.error(f"Error computing average probabilities: {e}")
        avg_probabilities = {emotion: 0 for emotion in CLASS_NAMES}

    # Format results for each emotion
    try:
        results_list = []
        for emotion in EMOTION_DESCRIPTIONS.keys():
            prob = avg_probabilities.get(emotion.lower(), 0)
            results_list.append({
                "name": emotion,
                "value": round(prob, 4),
                "description": EMOTION_DESCRIPTIONS.get(emotion, "")
            })
    except Exception as e:
        logging.error(f"Error formatting results list: {e}")
        results_list = []

    return results_list, best_face

def attendance_verification_from_face(best_face, folder_path: str, detector_backend: str = 'opencv'):
    """
    Compares the best detected face against stored images for verification.
    """
    if best_face is None:
        logging.error("No best face provided for attendance verification.")
        return None

    best_face_path = "best_face.jpg"
    verified_image_name = None

    try:
        if not os.path.exists(folder_path):
            raise Exception(f"Folder path does not exist: {folder_path}")
    except Exception as e:
        logging.error(f"Error accessing folder path {folder_path}: {e}")
        return None

    try:
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
                    logging.error(f"Error verifying {image_name}: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error listing images in folder {folder_path}: {e}")
        return None

    return verified_image_name

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main UI page
    """
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    FastAPI endpoint that:
      - Accepts a video file upload
      - Extracts faces, performs emotion analysis, and detects the best face
      - Verifies the best face against stored images for attendance
      - Updates MongoDB with the facial analysis and attendance result
    """
    try:
        # Save uploaded video file
        uploaded_video_path = "uploaded_face_video.mp4"
        try:
            contents = await file.read()
        except Exception as e:
            logging.error(f"Error reading uploaded file: {e}")
            raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

        try:
            with open(uploaded_video_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            logging.error(f"Error saving uploaded video: {e}")
            raise HTTPException(status_code=500, detail="Could not save the uploaded video.")

        # Process video for emotions & best face
        try:
            facial_results, best_face = process_video(uploaded_video_path)
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise HTTPException(status_code=500, detail="Error processing video.")

        # Verify attendance using best face
        try:
            attendance_image = attendance_verification_from_face(best_face, "../node_backend/uploads")
        except Exception as e:
            logging.error(f"Error verifying attendance: {e}")
            attendance_image = None

        attendance_result = {"verified": "Yes", "image": attendance_image} if attendance_image else "No verified image found"

        # Update MongoDB if a verified attendance image is found
        if attendance_image:
            try:
                user_id = attendance_image.rsplit('.', 1)[0]
            except Exception as e:
                logging.error(f"Error extracting user id from image name: {e}")
                user_id = None

            if user_id:
                try:
                    client = MongoClient(MONGO_URI)
                    db = client[DB_NAME]
                    collection = db[COLLECTION_NAME]
                except Exception as e:
                    logging.error(f"Error connecting to MongoDB: {e}")
                    raise HTTPException(status_code=500, detail="Error connecting to MongoDB.")

                update_data = {
                    "attendence": True,
                    "facial_analysis": facial_results
                }
                try:
                    result = collection.update_one({"_id": user_id}, {"$set": update_data})
                    if result.modified_count:
                        logging.info(f"Updated user {user_id} successfully in MongoDB.")
                    else:
                        logging.warning(f"No document found with _id: {user_id} to update.")
                except Exception as e:
                    logging.error(f"Error updating MongoDB for user {user_id}: {e}")
                    raise HTTPException(status_code=500, detail="Error updating MongoDB.")
                finally:
                    client.close()
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    finally:
        # Clean up uploaded video file
        try:
            if os.path.exists("uploaded_face_video.mp4"):
                os.remove("uploaded_face_video.mp4")
        except Exception as e:
            logging.error(f"Error deleting temporary video file: {e}")

    return {
        "Facial Analysis": facial_results,
        "Attendance": attendance_result
    }

# For development purposes, you can verify the application is working
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Facial Analysis API is running"}