from multiprocessing import Process, Array
import numpy as np
from sklearn import svm, preprocessing, feature_extraction
from sklearn.externals import joblib
import cv2
import glob
import imutils
from PIL import Image

size = 64, 64


def process_image(img):
    #img = img.resize(size, resample=Image.LANCZOS)
    img = img.convert('L')
    img = preprocessing.StandardScaler().fit(img).transform(img)
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


def load_samples_and_labels(samples_path, features):
    images = []
    labels = []
    feature_number = 0
    for sample_path in samples_path:
        for sample in glob.glob(sample_path):
            images.append(open_and_process_image(sample))
            labels.append(features[feature_number])
        feature_number += 1

    return images, labels


def get_frame(capture):
    ret, frame = capture.read()
    if not ret:
        raise IOError("No frame to read!")
    else:
        return frame


def detect_on_frame(frame, detector):
    img_to_predict = process_image(frame)
    return detector.predict(img_to_predict)


def change_frame(frame):
    # TODO
    return frame


def show_frame(frame):
    # TODO
    cv2.imshow('Frame', frame)
    return frame


def child_match_template(resized, template, ratio, interpolation, output_array):
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    output_array[interpolation * 4] = maxVal
    output_array[interpolation * 4 + 1] = maxLoc[0]
    output_array[interpolation * 4 + 2] = maxLoc[1]
    output_array[interpolation * 4 + 3] = ratio
    return 0


def get_matched_coordinates(image, template, interpolations):
    template_width = template.shape[1]
    template_height = template.shape[0]
    shared_array = Array('d', np.zeros((interpolations * 4, 1)))
    threads = []
    current_interpolation = 0
    for scale in np.linspace(0.2, 1.0, interpolations)[::-1]:
        resized = imutils.resize(image, width=int(image.shape[1] * scale))

        if resized.size[0] < template_height or resized.shape[1] < template_width:
            break

        ratio = image.shape[1] / float(resized.shape[1])
        p = Process(target=child_match_template, args=(resized, template, ratio, current_interpolation, shared_array,))
        threads.append(p)
        p.start()
        current_interpolation += 1

    for i in range(current_interpolation):
        threads[i].join()

    found = None

    for i in range(current_interpolation):
        if found is None or shared_array[i][0] > found[0]:
            place = i * 4
            found = (shared_array[place + 1], shared_array[place + 2], shared_array[place + 3])

    return found


def prepare_template_from_image(image):
    if image.shape[0] != image.shape[1]:
        image = crop_middle_square(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(gray, 50, 200)
    return template


def crop_middle_square(image):
    if image.shape[0] != image.shape[1]:
        w = image.shape[0]
        h = image.shape[1]
        if w > h:
            image = image[int((w-h)/2):int((w+h)/2), 0:h].copy()
        else:
            image = image[0:w, int((h-w)/2):int((h+w)/2)].copy()
    return image


if __name__ == '__main__':

    pred = []
    img = cv2.imread("template2.jpg", cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=256)
    img = prepare_template_from_image(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    images, labels = load_samples_and_labels(["kolka/*.jpg", "krzyzyki/*.jpg", "puste/*.jpg"], [1, -1, 0])

    machine = initialize_svn(images, labels, 'linear', 2, 1000)

    #pred.append(open_and_process_image('D:\MojeProjekty\PyCharm\VideoReader\IMG_20171114_110854.jpg'))

    #p1 = Process(target=predict, args=(machine, pred,))
    #p2 = Process(target=predict, args=(machine, pred,))

    #p1.start()
    #p2.start()
