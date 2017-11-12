from multiprocessing import Process, Lock,  Value, Array
import numpy as np
from sklearn import svm, preprocessing, feature_extraction
from sklearn.externals import joblib
import cv2
import glob
import scipy.misc
from PIL import Image

Scaler = preprocessing.StandardScaler()
size = 128, 128


def get_frame(capture):
    ret, frame = capture.read()
    if not ret:
        raise IOError("No frame to read!")
    else:
        return frame


def detect_on_frame(frame, machine):
    img_to_predict = process_image(frame)
    return machine.predict(img_to_predict)


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


def initialize_svn(samples, features, kernel, gamma, cache):
    svc = svm.SVC(kernel=kernel, gamma=gamma, cache_size=cache)
    svc.fit(samples, features)
    return svc


def predict(svn, prediction):
    print(svn.predict(prediction))


def load_samples_and_labels(samples, features):
    images = []
    labels = []
    for sample, feature in samples, features:
        images.append(open_and_process_image(sample))
        labels.append(feature)

    return images, labels


if __name__ == '__main__':

    pred = []

    images = []
    labels = []
    for file in glob.glob("plearn/*.jpg"):
        images.append(open_and_process_image(file))
        labels.append('yes')

    for file in glob.glob("pnotlearn/*.jpg"):
        images.append(open_and_process_image(file))
        labels.append('no')

    machine = initialize_svn(images, labels, 'linear', 2, 1000)

    pred.append(open_and_process_image('1.jpg'))

    pred.append(open_and_process_image('n1.jpg'))

    p1 = Process(target=predict, args=(machine, pred,))
    p2 = Process(target=predict, args=(machine, pred,))

    p1.start()
    p2.start()
