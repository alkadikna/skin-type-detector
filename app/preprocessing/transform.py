import cv2
from mtcnn import MTCNN

IMAGE_SIZE = (224, 224)
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