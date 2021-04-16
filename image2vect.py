import os
import numpy as np
from yolo_face import YOLO
import tensorflow as tf
from sklearn.preprocessing import normalize

FACENET_MODEL_DIR = './facenet/facenet_keras.h5'

class ImageVector(object):
    def __init__(self):
        self.yolo_model = YOLO()
        self.facenet_model = tf.keras.models.load_model(FACENET_MODEL_DIR)
    
    def vectorize_image(self, image):
        """
        Takes an image and calculates bounding boxes on faces.
        Returns an image embedding vector normalized.
        """
        # Get face boxes from YOLO
        image, out_boxes = self.yolo_model.detect_image(image)

        if (out_boxes.size != 0):
            # Crop face boxes for FaceNet input
            image_list = self.crop_image(image, out_boxes)

            # Convert from int to float32
            image_list = np.array(image_list, dtype='float32')
            # image_list = image_list.astype('float32')
            # Normalizing input data for FaceNet
            image_list /= 255.0

            # Get embedding vector from FaceNet
            embedding_vector = self.facenet_model.predict(image_list)

            # Normalize embedding vector
            embedding_vector = normalize(embedding_vector, norm='l2')
        # When no faces are detected
        else:
            embedding_vector = np.full((1,128), 0)

        return embedding_vector

    def crop_image(self, image, out_boxes):
        """ Takes an image and returns a list of cropped images"""
        crop_size = 160 # Not sure why, but error occurs if size is not this
        image_list = []
        for box in out_boxes:
            # Get box dimensions, code taken from __init__.py (yolo.py)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

             # Crop box dimension
            cropped_img = image.crop((left, top, right, bottom))
            # Resize cropped image to crop_size
            cropped_img = cropped_img.resize((crop_size, crop_size)) 

            # Reshape for 3 color channels
            cropped_img_list = np.array(list(cropped_img.getdata())).reshape((crop_size, crop_size, 3)) 
            image_list.append(cropped_img_list)

        return image_list
