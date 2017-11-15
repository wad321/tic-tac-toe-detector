from multiprocessing import Process, Array
import numpy as np
from sklearn import svm, preprocessing, feature_extraction
from sklearn.externals import joblib
import cv2
import glob
import imutils
from PIL import Image

size = 64, 64


def process_frame(frame, changetograyscale):
    frame = crop_middle_square(frame)
    frame = imutils.resize(frame, width=size[0], inter=cv2.INTER_LANCZOS4)
    if changetograyscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = cv2.Canny(frame, 80, 200)
    # cv2.imshow('image', frame)
    # cv2.waitKey(0)
    frame = frame.astype(np.float64)
    frame = preprocessing.StandardScaler().fit(frame).transform(frame)
    frame = frame.flatten()
    return frame


def open_and_process_image(filename):
    image_to_process = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return process_frame(image_to_process, True)


def initialize_svn(samples, features, c, gamma):
    svc = svm.SVC(kernel='rbf', C=c, cache_size=1000)
    svc.fit(samples, features)
    return svc


def predict(svn, prediction):
    array = []
    for obj in svn.predict(prediction):
        array.append(obj)
    return array


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
    img_to_predict = process_frame(frame, False)
    return detector.predict(img_to_predict)


def change_frame(frame):
    # TODO
    return frame


def show_frame(frame):
    # TODO
    cv2.imshow('Frame', frame)
    return frame


def child_match_template(resized, template, ratio, interpolation, output_array):
    edged = cv2.Canny(resized, 80, 200)
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

        if resized.shape[0] < template_height or resized.shape[1] < template_width:
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
        place = i * 4
        if found is None or shared_array[place] > found[0]:
            found = (shared_array[place + 1], shared_array[place + 2], shared_array[place + 3])

    return found


def prepare_template_from_image(image, width):
    image = imutils.resize(image, width=width)
    if image.shape[0] != image.shape[1]:
        image = crop_middle_square(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(gray, 100, 200)
    #cv2.imshow('template', template)
    #cv2.waitKey(0)
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


def get_nine_images(image, start, end):
    nine_images = []
    coords = np.zeros((2, 4))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for i in range(2):
        coords[i] = [start[i], int((2 * start[i] + end[i])/3), int((start[i] + 2 * end[i])/3), end[i]]
    coords = coords.astype(np.int)

    for y in range(3):
        for x in range(3):
            crop = image[coords[1][x]:coords[1][x+1], coords[0][y]:coords[0][y+1]].copy()
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            nine_images.append(process_frame(crop, False))

    return nine_images


def match_two_templates(image, start, end, circle, cross, threshold):
    array = []
    coords = np.zeros((2, 4))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for i in range(2):
        coords[i] = [start[i], int((2 * start[i] + end[i]) / 3), int((start[i] + 2 * end[i]) / 3), end[i]]
    coords = coords.astype(np.int)

    for y in range(3):
        for x in range(3):
            crop = image[coords[1][x]:coords[1][x+1], coords[0][y]:coords[0][y+1]].copy()
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            canny = cv2.Canny(crop, 100, 200)
            cr = cv2.matchTemplate(canny, cross)
            ci = cv2.matchTemplate(canny, circle)
            (_, maxCrossVal, _, _) = cv2.minMaxLoc(cr)
            (_, maxCircleVal, _, _) = cv2.minMaxLoc(ci)

            if maxCrossVal < threshold and maxCircleVal < threshold:
                array.append(0)
            elif maxCircleVal > maxCrossVal:
                array.append(1)
            else:
                array.append(-1)

    return array


if __name__ == '__main__':

    pred = []
    img = cv2.imread("template1v2.jpg", cv2.IMREAD_COLOR)
    img2 = img.flatten()
    testimage = cv2.imread("plearn/2.jpg", cv2.IMREAD_COLOR)
    testimage = imutils.resize(testimage, width=800)
    template = prepare_template_from_image(img, 256)

    found = get_matched_coordinates(testimage, template, 20)
    (maxLoc0, maxLoc1, r) = found
    startXY = (int(maxLoc0 * r), int(maxLoc1 * r))
    endXY = (int((maxLoc0 + template.shape[1]) * r), int((maxLoc1 + template.shape[0]) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(testimage, startXY, endXY, (0, 0, 255), 2)
    cv2.imshow("Image", testimage)
    cv2.waitKey(0)

    nineobjs = get_nine_images(testimage, startXY, endXY)

    images, labels = load_samples_and_labels(["kolka/*.jpg", "krzyzyki/*.jpg", "puste/*.jpg"], [1, -1, 0])

    c = 2**(-5)
    gamma = 2**(-15)
    answer = [-1, 0, 0, 0, 1, 0, 0, 0, -1]

    while gamma <= 2 ** 2:
        while c <= 2 ** 3:
            goodhits = 0
            machine = initialize_svn(images, labels, c, gamma)

            array = predict(machine, nineobjs)

            for i in range(9):
                if answer[i] == array[i]:
                    goodhits += 1
            print(goodhits)
            if goodhits >= 5:
                output = open('output.txt', mode='a', encoding='utf-8')
                print('HAHAHAHHA!')
                line = str(str(goodhits) + " : " + str(gamma) + " " + str(c) + " " + " ".join(str(x) for x in array))
                output.write(line + "\n")
                output.close()

            #predict(machine, nineobjs, gamma, c)
            c = c * 2
        c = 2 ** (-5)
    gamma = gamma * 2


    #pred.append(open_and_process_image('D:\MojeProjekty\PyCharm\VideoReader\IMG_20171114_110854.jpg'))
