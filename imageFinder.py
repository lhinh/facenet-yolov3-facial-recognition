import numpy as np
from image2vect import ImageVector
from os import listdir
from keras.models import load_model
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

FACENET_MODEL_DIR = './facenet/facenet_keras.h5'
DATA_DIR = './Data/subset/'

DATA_PARENT_DIR = './Data/'
CELEB_ID_DIR = DATA_PARENT_DIR + 'selected_ids.txt'
CELEB_IMG_ID_DIR = DATA_PARENT_DIR + 'identity_CelebA.txt'

class ImageFinder(object):
    def __init__(self):
        self.image_to_vector = ImageVector()
        self.yolo_model = self.image_to_vector.yolo_model
        self.facenet_model = load_model(FACENET_MODEL_DIR)
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def load_data(self):
        image_list = []
        # Store PIL Images from entire subset data into list
        for file_name in listdir(DATA_DIR):
            image_list.append(Image.open(DATA_DIR + file_name))
        return image_list

    def get_file_id_dict(self):
        # Load all filename/ID mappings into Python as dict
        celeb_img_id_dict = {}
        with open(CELEB_IMG_ID_DIR) as f:
            celeb_img_id_list = f.read().splitlines()

        # Make keys : file_names and values : ID
        for line in celeb_img_id_list:
            split_kv = line.split(' ')
            celeb_img_id_dict[split_kv[0]] = int(split_kv[1])
        return celeb_img_id_dict
        
    def calc_euclidean_distance(self, image, threshold=1.0):
        """
        Takes a PIL Image and calculates euclidean distances against
        the rest of the data set
        """
        # Load 1,200 images
        image_list = self.load_data()

        # Get file name and id dictionary
        celeb_img_id_dict = self.get_file_id_dict()

        # Get celebrity ID from file name
        file_name = image.filename.split('./Data/subset/')[1] # File name only
        file_id = celeb_img_id_dict[file_name]

        # Get input embedding vector
        target_embedding_vector = self.image_to_vector.vectorize_image(image)

        matching_images = []
        # Get the embedding vectors of other images and compare iteratively
        for subset_image in image_list:
            compare_embedding_vector = self.image_to_vector.vectorize_image(subset_image)
            distances = euclidean_distances(target_embedding_vector, compare_embedding_vector)
            distances = distances[0] # Convert to a single list rather than list of list

            # Retrieving ID for metrics
            subset_file_name = subset_image.filename.split('./Data/subset/')[1]
            subset_img_id = celeb_img_id_dict[subset_file_name]

            print(distances)
            for dist in distances:
                if dist < threshold:
                    matching_images.append(subset_image.filename)

                if file_id == subset_img_id and dist < threshold: # True positive
                    self.true_positive += 1
                elif file_id != subset_img_id and dist < threshold: # False positive
                    self.false_positive += 1
                elif file_id == subset_img_id and dist >= threshold: # False negative
                    self.false_negative += 1

        return matching_images

    # Precision = true positives / (true positives + false positives)
    def get_precision(self):
        # denominator check
        if self.true_positive + self.false_positive == 0:
            return 0
        else:
            return self.true_positive / (self.true_positive + self.false_positive)

    # Recall = true positives / (true positives + false negatives)
    def get_recall(self):
        if self.true_positive + self.false_negative == 0:
            return 0
        else:
            return self.true_positive / (self.true_positive + self.false_negative)
