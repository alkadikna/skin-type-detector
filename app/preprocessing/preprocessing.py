import os
import numpy as np
import cv2
from mtcnn import MTCNN
import random

# Parameter dasar
IMAGE_SIZE = (224, 224)
DATA_DIR = "app/data"
CLASSES = os.listdir(DATA_DIR)

detector = MTCNN()

def detect_and_crop_face(image):
    results = detector.detect_faces(image)
    if not results:
        return None

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, IMAGE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    return face

def preprocess_image_for_prediction(img_path):
    """
    Fungsi preprocessing satu gambar (misalnya dipakai saat user upload)
    Return: image dalam bentuk array (1, 224, 224, 3) siap untuk model.predict()
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Gagal membaca gambar: {img_path}")
        return None

    face = detect_and_crop_face(img)
    if face is not None:
        face = np.expand_dims(face, axis=0)  # bentuk jadi (1, 224, 224, 3)
        print(f"[INFO] Wajah terdeteksi: {img_path}")
        return face
    else:
        print(f"[ERROR] Wajah tidak terdeteksi di: {img_path}")
        return None

def load_dataset():
    images = []
    labels = []

    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, label)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[SKIP] Gagal membaca gambar: {img_path}")
                    continue

                face = detect_and_crop_face(img)
                if face is not None:
                    face = augment_image(face)  # Menambahkan augmentasi di sini
                    images.append(face)
                    labels.append(idx)
                    print(f"[INFO] Wajah terdeteksi: {img_path}")
                else:
                    print(f"[SKIP] Tidak ada wajah terdeteksi: {img_path}")

    return np.array(images), np.array(labels)

def augment_image(image):
    """
    Fungsi augmentasi gambar (misalnya rotasi, flipping, dsb)
    """
    # Rotasi acak antara 0, 90, 180, dan 270 derajat
    rotate_angle = random.choice([0, 90, 180, 270])
    augmented_image = cv2.rotate(image, rotate_angle)
    
    # Flipping acak (horizontal atau vertikal)
    if random.choice([True, False]):
        augmented_image = cv2.flip(augmented_image, 1)  # Flipping horizontal
    else:
        augmented_image = cv2.flip(augmented_image, 0)  # Flipping vertical

    # Perubahan kecerahan (menambah atau mengurangi brightness)
    brightness_factor = random.uniform(0.7, 1.3)
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=brightness_factor, beta=0)

    return augmented_image

if __name__ == "__main__":
    # Load dan proses dataset
    X, y = load_dataset()
    print(f"Total wajah terdeteksi: {len(X)}")
    print(f"Bentuk X: {X.shape}, y: {y.shape}")

    # Simpan hasil
    np.save("app/fix-dataset/preprocessed_faces.npy", X)
    np.save("app/fix-dataset/preprocessed_labels.npy", y)
    print("[INFO] Dataset telah disimpan.")