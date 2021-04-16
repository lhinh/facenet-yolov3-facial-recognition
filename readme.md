# Facial Recognition with YoloV3 and FaceNet

The purpose of this project is to use pre-trained models of YoloV3 and FaceNet to identify celebrity faces. This involves a bit of data wrangling to only select a subset of images (selected_ids.txt) to be used for facial recognition.

## Project setup

Please download the files from [here](https://drive.google.com/drive/folders/19chVP0F1Yqxdu9jscI12ocFXNQWzTudF?usp=sharing) and save in the following directories:

- ./Data/identity_CelebA.txt
- ./Data/selected_id.txt
- ./yolo_face/YOLO_Face.h5
- ./yolo_face/yolo_anchors.txt
- ./yolo_face/face_classes.txt
- ./facenet/facenet_keras.h5

Also please download the [dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28) and extract in this directory:

- ./Data/img_celeba

Next, create the following directory:

- ./Data/subset/

Finally, run process-data.py to create a subset of images from selected_id.txt

## Requirements

- Python       == 3.6.8 64-bit <- This is important!
- scikit-learn == 0.24.1
- pillow       == 8.2.0

### The following will not install properly if Python is not 64-bit

- tensorflow == 2.4.1
- Keras      == 2.4.3

## References
- Pre-trained YoloV3 by Thanh Nguyen: https://github.com/sthanhng/yoloface
- Pre-trained FaceNet model by Hiroki Taniai: https://github.com/nyoki-mtl
- Dive Really Deep into YOLO v3: A Beginner’s Guide: https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e
- Introduction to FaceNet: A Unified Embedding for Face Recognition and Clustering: https://medium.com/analytics-vidhya/introduction-to-facenet-a-unified-embedding-for-face-recognition-and-clustering-dbdac8e6f02
- How to Perform Object Detection With YOLOv3 in Keras https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
- How to Develop a Face Recognition System Using FaceNet in Keras: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/