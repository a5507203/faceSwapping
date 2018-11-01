# import dlib
# import numpy as np
import face_recognition


# import face_recognition_models

class FaceFilter():
    def __init__(self, reference_file_path, threshold=0.6):
        image = face_recognition.load_image_file(reference_file_path)
        self.encoding = face_recognition.face_encodings(image)[
            0]  # Note: we take only first face, so the reference file should only contain one face.
        self.threshold = threshold

    def check(self, detected_face):
        encodings = face_recognition.face_encodings(
            detected_face.image)  # we could use detected landmarks, but I did not manage to do so. TODO The copy/paste below should help
        if encodings is not None and len(encodings) > 0:
            score = face_recognition.face_distance([self.encoding], encodings[0])
            print(score)
            return score <= self.threshold
        else:
            print("No face encodings found")
            return False
