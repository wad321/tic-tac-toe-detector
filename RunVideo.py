from multiprocessing import Process, Array, Value
from multiprocessing.pool import ThreadPool
import numpy as np
from sklearn import svm, preprocessing
import cv2
import glob
import imutils
from collections import deque


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


def open_and_process_image(filename):
    image_to_process = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return process_frame(image_to_process, True)


def process_frame(f_frame, changetograyscale):
    f_frame = crop_middle_square(f_frame)
    f_frame = imutils.resize(f_frame, width=size[0], inter=cv2.INTER_LINEAR)

    if changetograyscale:
        f_frame = cv2.cvtColor(f_frame, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('image', f_frame)
    # cv2.waitKey(0)

    if svn_format:
        f_frame = f_frame.astype(np.float64)
        f_frame = preprocessing.StandardScaler().fit(f_frame).transform(f_frame)
        f_frame = f_frame.flatten()
    else:
        f_frame = cv2.Canny(f_frame, 80, 200)

    return f_frame


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
            found = (shared_array[place], shared_array[place + 1], shared_array[place + 2], shared_array[place + 3])

    return found


def prepare_template_from_image(image, width):
    image = imutils.resize(image, width=width)
    if image.shape[0] != image.shape[1]:
        image = crop_middle_square(image)
    f_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_template = cv2.Canny(f_gray, 100, 200)
    # cv2.imshow('template', template)
    # cv2.waitKey(0)
    return f_template


def crop_middle_square(image):
    if image.shape[0] != image.shape[1]:
        w = image.shape[0]
        h = image.shape[1]
        if w > h:
            image = image[int((w-h)/2):int((w+h)/2), 0:h].copy()
        else:
            image = image[0:w, int((h-w)/2):int((h+w)/2)].copy()

    return image


def match_two_templates(image, start, end, circle, cross):
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
            canny = cv2.Canny(crop, 80, 200)
            cr = cv2.matchTemplate(canny, cross,  cv2.TM_CCOEFF)
            ci = cv2.matchTemplate(canny, circle,  cv2.TM_CCOEFF)
            (_, maxCrossVal, _, _) = cv2.minMaxLoc(cr)
            (_, maxCircleVal, _, _) = cv2.minMaxLoc(ci)

            if maxCrossVal < template_threshold and maxCircleVal < template_threshold:
                array.append(0)
            elif maxCircleVal > maxCrossVal:
                array.append(1)
            else:
                array.append(-1)

    return array


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
            # cv2.imshow('crop', crop)
            # cv2.waitKey(0)
            nine_images.append(process_frame(crop, False))

    return nine_images


def second_process(f_frame, number):
    f_template = main_template.copy()
    f_gray = cv2.cvtColor(f_frame, cv2.COLOR_RGB2GRAY)
    match = get_matched_coordinates(f_gray, f_template, match_template_interpolations)
    if match[0] > template_threshold:
        (_, maxLoc0, maxLoc1, r) = match
        startxy = (int(maxLoc0 * r), int(maxLoc1 * r))
        endxy = (int((maxLoc0 + f_template.shape[1]) * r), int((maxLoc1 + f_template.shape[0]) * r))

        # if svn_format:
        #    nine_images = get_nine_images(f_gray, startxy, endxy)
        # else:
        #    match_two_templates()
        cv2.rectangle(f_frame, startxy, endxy, (0, 0, 255), 2)
        place_to_draw = [startxy, endxy]
    else:
        place_to_draw = [(-1, -1), (-1, -1)]

    return f_gray, place_to_draw


size = 64, 64
svn_format = False
template_threshold = 3200000.0
frames_between_detection = 10
match_template_interpolations = 20

img = cv2.imread("template1v2.jpg", cv2.IMREAD_COLOR)
main_template = prepare_template_from_image(img, 256)


if __name__ == '__main__':
    if svn_format:
        images, labels = load_samples_and_labels(["kolka/*.jpg", "krzyzyki/*.jpg", "puste/*.jpg"], [1, -1, 0])

    cross = cv2.imread("template1v2.jpg", cv2.IMREAD_COLOR)
    circle = cv2.imread("template1v2.jpg", cv2.IMREAD_COLOR)
    templates = [prepare_template_from_image(cross, 64), prepare_template_from_image(circle, 64)]

    matched_place = Array('d', np.zeros((4, 1)))

    change_on_frame = Value('i', lock=True)
    change_on_frame = 0

    frame_number = 0

    where_draw = [(-1, -1), (-1, -1)]

    cap = cv2.VideoCapture("saint.mp4")
    if not cap.isOpened():
        print("is not opened")
        cv2.VideoCapture.open()

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    # MAIN EVENT TIME!
    while cap.isOpened():
        while len(pending) > 0 and pending[0].ready():

            res, where_draw = pending.popleft().get()
            cv2.imshow('tic-tac-toe', res)
            cv2.waitKey(30)

        ret, frame = cap.read()

        if len(pending) < threadn and frame_number == frames_between_detection:
            task = pool.apply_async(second_process, (frame.copy(), frame_number))
            pending.append(task)
            frame_number = 0

        if where_draw[0][0] > 0:
            cv2.rectangle(frame, where_draw[0], where_draw[1], (0, 0, 255), 2)
            cv2.imshow('tic-tac-toe', frame)
        else:
            cv2.imshow('tic-tac-toe', frame)
        ch = 0xFF & cv2.waitKey(30)
        if ch == ord('q') or ch == 27:
            break

        if frame_number > 10:
            frame_number = 0
        else:
            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
