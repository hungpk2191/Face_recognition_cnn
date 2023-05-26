import os
import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
import cv2
from keras.layers import Input
from keras import backend as K
from keras_openface import utils
from keras_openface.utils import LRN2D
from create_embeddings import create_model
import dlib
import time
Input = Input(shape=(96, 96, 3))
model = create_model(Input)
url_camDahua = "rtsp://admin:Camt3hvn@192.168.2.191:554/cam/realmonitor?channel=1&subtype=0"
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

weights = utils.weights
weights_dict = utils.load_weights()

for name in weights:
    print("Setting........... ")
    if model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])
    elif model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])


def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.around(np.transpose(img, (0, 1, 2))/255.0, decimals=12)
    img_array = np.array([img])

    embedding = model.predict_on_batch(img_array)
    return embedding


def recognize_face(face_image, embeddings, model):

    face_embedding = image_to_embedding(face_image, model)

    min_dist = 150
    Name = None
    for (name, embedding) in embeddings.items():

        dist = np.linalg.norm(face_embedding-embedding)
        if dist < min_dist:
            min_dist = dist
            Name = name
    if min_dist <= 0.75:
        return str(Name)
    else:
        return None


def process_frame(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    for face in faces:
        left = face.rect.left()
        top = face.rect.top()
        right = face.rect.right()
        bottom = face.rect.bottom()
        facee = frame[top:bottom, left:right]
        identity = recognize_face(facee, embeddings, model)
        if identity is not None:
            print(str(identity).title())


def recognize_faces(embeddings):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    start_time = time.time()


    while True:
        ret, image = cap.read()
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 2:
            process_frame(image)
            start_time = current_time
            
        # cv2.imshow("Face Recognizer", image)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # exit on q
            break
    cap.release()
    cv2.destroyAllWindows()

def load_embeddings():
    input_embeddings = {}
    embedding_file = np.load('embeddings.npy', allow_pickle=True)
    for k, v in embedding_file[()].items():
        print(type(v))
        input_embeddings[k] = v

    return input_embeddings


embeddings = load_embeddings()
print(embeddings)
recognize_faces(embeddings)
