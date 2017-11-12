from multiprocessing import Process, Lock,  Value, Array
import numpy as np
from sklearn import svm, datasets, preprocessing, feature_extraction
import cv2
import glob
from skimage import io
import scipy.misc
from PIL import Image

Scaler = preprocessing.StandardScaler()
size = 128, 128
__all__ = [Process, Lock,  Value, Array, np, svm, datasets, preprocessing, feature_extraction, cv2, io, glob, scipy.misc]


def get_frame(capture):
    ret, frame = capture.read()
    if not ret:
        raise IOError("No frame to read!")
    else:
        return frame


def detect_on_frame(frame, machine):
    # TODO
    return frame


def change_frame(frame):
    # TODO
    return frame


def show_frame(frame):
    # TODO
    cv2.imshow('Frame', frame)
    return frame


def process_image(img):
    img = img.resize(size, resample=Image.LANCZOS)
    img = img.convert('L')
    img = Scaler.fit(img).transform(img)
    img = img.flatten()
    return img


def open_and_process_image(filename):
    image_to_process = Image.open(filename)
    return process_image(image_to_process)


def initialize_svm(kernel, gamma):
    images = []
    features = []
    svc = svm.SVC(kernel=kernel, gamma=gamma)
    for file in glob.glob("plearn/*.jpg"):
        images.append(open_and_process_image(file))
        features.append(1)

    for file in glob.glob("pnotlearn/*.jpg"):
        images.append(open_and_process_image(file))
        features.append(0)

    svc.fit(images, features)
    return svc


if __name__ == '__main__':

    pred = []
    machine = initialize_svm('linear', 1)

    pred.append(open_and_process_image('1.jpg'))

    pred.append(open_and_process_image('n1.jpg'))

    print(machine.predict(pred))
