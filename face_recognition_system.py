"""
Face detection evaluation based on facenet with sample from VGGFcce2
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


# face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)
# face_embedding = InceptionResnetV1(pretrained='vggface2').eval()


class FaceRecognitionSystem:
    def __init__(self, face_detector, face_embedding, users_images_folder):
        self.face_detector = face_detector
        self.face_embedding = face_embedding  # THE DISTINGUISHER

        users_images = datasets.ImageFolder(users_images_folder)
        loader = DataLoader(users_images, collate_fn=lambda x: x[0])
        idx_to_name = {i: c for c, i in users_images.class_to_idx.items()}

        self.users = dict()
        max_users = 700
        i = 0
        for img, idx in loader:
            if i > max_users:
                break
            face, prob = face_detector(img, return_prob=True)
            if face is not None and prob > 0.90:
                self.users[idx] = {
                    'embedding': face_embedding(face.unsqueeze(0)),
                    'name': idx_to_name[idx],
                    'face': face
                }
            else:
                self.users[idx] = None
            i += 1

    def get_invalid_users(self):
        return [user for user in self.users if self.users[user] is not None]

    def recognize_by_image(self, img, similarity_threshold=None, ret_order_of_users=False):
        if isinstance(img, str):
            img = Image.open(img)
        face, prob = self.face_detector(img, return_prob=True)
        emb = self.face_embedding(face.unsqueeze(0)).detach()
        dists = []
        min_distance = None
        min_idx_distance = None
        for idx, user_info in self.users.items():
            if user_info is None:
                continue
            dist = torch.dist(emb, user_info['embedding']).item()
            dists.append((dist, user_info['name']))
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_idx_distance = idx

        if ret_order_of_users:
            dists = sorted(dists, key=lambda x: x[0])
            return [x[1] for x in dists]

        if similarity_threshold is None or min_distance < similarity_threshold:
            return self.users[min_idx_distance]

        return None


def create_face_recognition_system(users_images_folder='./GYM_MEMBERS', pretrained_on='vggface2'):
    face_recognition_system = FaceRecognitionSystem(
        face_detector=MTCNN(image_size=240, margin=0, min_face_size=20),
        face_embedding=InceptionResnetV1(pretrained=pretrained_on).eval(),
        # FOR BLACK BOX USE pretrained='casia-webface'
        users_images_folder=users_images_folder
    )

    return face_recognition_system
