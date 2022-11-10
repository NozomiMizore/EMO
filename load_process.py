import pandas as pd
import cv2
import numpy as np

dataset_path = 'data/fer2013/fer2013.csv'
image_size = (48,48)

def load_dataset():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_seq in pixels:
        face = [int(pixel) for pixel in pixel_seq.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

def process_data(x, v=True):
    x = x.astype('float32')
    x /= 255.0
    if v:
        x = x - 0.5
        x = x * 2.0
    return x