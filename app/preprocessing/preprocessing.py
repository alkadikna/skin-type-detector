import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameter dasar
IMAGE_SIZE = (224, 224)
DATA_DIR = "data/face_skin_type"
CLASSES = os.listdir(DATA_DIR)

# Setup Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_face_mediapipe(image):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            # Ambil bounding box pertama
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            # Pastikan bounding box valid
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_min + box_width, w)
            y_max = min(y_min + box_height, h)

            face_img = image[y_min:y_max, x_min:x_max]
            return face_img
        else:
            return None

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    face = detect_and_crop_face_mediapipe(img)
    
    if face is None:
        print(f"[SKIP] Tidak ada wajah terdeteksi: {img_path}")
        return None

    face = cv2.resize(face, IMAGE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    return face

def load_dataset():
    images = []
    labels = []

    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, label)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = load_and_preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(idx)

    return np.array(images), np.array(labels)

# Load dataset
X, y = load_dataset()
print(f"Total data wajah terdeteksi: {len(X)}")
print(f"Bentuk X: {X.shape}, y: {y.shape}")

# Augmentasi opsional
augment = True
if augment:
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X)

# Simpan hasil preprocessing
np.save("preprocessed_faces_mediapipe.npy", X)
np.save("preprocessed_labels_mediapipe.npy", y)
