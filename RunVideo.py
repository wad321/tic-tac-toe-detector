from multiprocessing import Process, Array
from multiprocessing.pool import ThreadPool
import numpy as np
from sklearn import svm, preprocessing
import cv2
import glob
import imutils
from collections import deque


def load_samples_and_labels(samples_path, features):
    f_images = []
    f_labels = []
    feature_number = 0
    for sample_path in samples_path:
        for sample in glob.glob(sample_path):
            f_images.append(open_and_process_image(sample))
            f_labels.append(features[feature_number])
        feature_number += 1

    return f_images, f_labels


def open_and_process_image(filename):
    image_to_process = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return process_frame(image_to_process, True)


def predict(svn, prediction):
    array = []
    for obj in svn.predict(prediction):
        array.append(obj)
    return array


def initialize_svn(samples, features, c, gamma):
    svc = svm.SVC(kernel='rbf', C=c, gamma=gamma, cache_size=1000)
    svc.fit(samples, features)
    return svc


def process_frame(f_frame, changetograyscale):
    f_frame = crop_middle_square(f_frame)
    f_frame = imutils.resize(f_frame, width=size[0], inter=cv2.INTER_LINEAR)

    if changetograyscale:
        f_frame = cv2.cvtColor(f_frame, cv2.COLOR_RGB2GRAY)

    f_frame = cv2.Canny(f_frame, 80, 200)
    # cv2.imshow('image', f_frame)
    # cv2.waitKey(0)

    if svn_format:
        f_frame = f_frame.astype(np.float64)
        f_frame = preprocessing.StandardScaler().fit(f_frame).transform(f_frame)
        f_frame = f_frame.flatten()

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

    for inter in range(current_interpolation):
        threads[inter].join()

    found = None

    for inter in range(current_interpolation):
        place = inter * 4
        if found is None or shared_array[place] > found[0]:
            found = (shared_array[place], shared_array[place + 1], shared_array[place + 2], shared_array[place + 3])
    return found


def prepare_template_from_image(image, width):
    image = imutils.resize(image, width=width)
    if image.shape[0] != image.shape[1]:
        image = crop_middle_square(image)
    f_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_template = cv2.Canny(f_gray, 100, 150)
    # cv2.imshow('template', f_template)
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


def match_two_templates(image, f_coords, templates):
    array = []

    for y in range(3):
        for x in range(3):
            crop = image[f_coords[1][x]:f_coords[1][x+1], f_coords[0][y]:f_coords[0][y+1]].copy()
            resized = imutils.resize(crop, width=circle_cross_size, inter=cv2.INTER_LINEAR)
            # cv2.imshow('crop', crop)
            # cv2.waitKey(0)
            canny = cv2.Canny(resized, 80, 200)
            cross = cv2.matchTemplate(canny, templates[0],  cv2.TM_CCOEFF)
            circle = cv2.matchTemplate(canny, templates[1],  cv2.TM_CCOEFF)
            (_, maxCrossVal, _, _) = cv2.minMaxLoc(cross)
            (_, maxCircleVal, _, _) = cv2.minMaxLoc(circle)

            print('(y, x) : Kolko/Krzyzyk  -> (', y, x, ') : ', maxCrossVal, maxCircleVal)
            if maxCrossVal < secondary_threshold and maxCircleVal < secondary_threshold:
                array.append(0)
            elif maxCircleVal > maxCrossVal:
                array.append(1)
            else:
                array.append(-1)

    return array


def get_nine_images(image, f_coords):
    nine_images = []

    for y in range(3):
        for x in range(3):
            crop = image[f_coords[1][x]:f_coords[1][x+1], f_coords[0][y]:f_coords[0][y+1]].copy()
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

        place_to_draw = [startxy, endxy]

        if enable_cricle_cross_detection:
            coords = np.zeros((2, 4))
            for it in range(2):
                coords[it] = [startxy[it], int((2 * startxy[it] + endxy[it]) / 3),
                              int((startxy[it] + 2 * endxy[it]) / 3), endxy[it]]
            coords = coords.astype(np.int)

            if svn_format:
                nine_images = get_nine_images(f_gray, coords)
                places = predict(svn_machine, nine_images)
            else:
                places = match_two_templates(f_gray, coords, (cross_template, circle_template))

            to_add = []
            for y in range(3):
                for x in range(3):
                    to_add.append((places[x+y], (coords[0][y], coords[1][x]), (coords[0][y+1], coords[1][x+1])))

            place_to_draw += to_add

    else:
        place_to_draw = [(-1, -1), (-1, -1)]

    return place_to_draw


# Main options
template_size = 96
video_size_width = 600
template_threshold = 7500000.0
frames_between_detection = 140
match_template_interpolations = 15
enable_cricle_cross_detection = False

# Circle and cross detection options
svn_format = False
size = 32, 32
circle_cross_size = 64
secondary_threshold = 6000000.0


img = cv2.imread("templates/template1v2.jpg", cv2.IMREAD_COLOR)
main_template = prepare_template_from_image(img, template_size)

if svn_format:
    images, labels = load_samples_and_labels(["kolka/*.jpg", "krzyzyki/*.jpg", "puste/*.jpg"], [1, -1, 0])
    svn_machine = initialize_svn(images, labels, 1, 0.02)
else:
    circle_template = prepare_template_from_image(
        cv2.imread("templates/circle_template.jpg", cv2.IMREAD_COLOR), circle_cross_size)
    cross_template = prepare_template_from_image(
        cv2.imread("templates/cross_template.jpg", cv2.IMREAD_COLOR), circle_cross_size)


if __name__ == '__main__':
    frame_number = 0

    where_draw = [(-1, -1), (-1, -1)]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("is not opened")
        cv2.VideoCapture.open()

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    process_finished = False

    # MAIN EVENT TIME!
    while cap.isOpened():
        while len(pending) > 0 and pending[0].ready():
            where_draw = pending.popleft().get()

        ret, frame = cap.read()
        frame = imutils.resize(frame, width=video_size_width, inter=cv2.INTER_LINEAR)

        if len(pending) < threadn and frame_number == frames_between_detection:
            task = pool.apply_async(second_process, (frame.copy(), frame_number))
            pending.append(task)
            frame_number = 0

        if where_draw[0][0] > 0:
            if enable_cricle_cross_detection:
                for i in range(2, 11):
                    cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)
                    if where_draw[i][0] == -1:
                        where_circle = (int((where_draw[i][1][0] + where_draw[i][2][0]) / 2),
                                        int((where_draw[i][1][1] + where_draw[i][2][1]) / 2))
                        radius = int(0.75 * (where_draw[i][2][0] -
                                             int((where_draw[i][1][0] + where_draw[i][2][0]) / 2)))
                        cv2.circle(frame, where_circle, radius, (0, 255, 0), 2)
                    elif where_draw[i][0] == 1:
                        line2 = ((where_draw[i][1][0] - 10, where_draw[i][2][1] + 10),
                                 (where_draw[i][1][1] + 10, where_draw[i][2][0] - 10))

                        cv2.line(frame, where_draw[i][1], where_draw[i][2], (0, 255, 0), 2)
                        cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)

            cv2.rectangle(frame, where_draw[0], where_draw[1], (0, 0, 255), 2)
            cv2.imshow('tic-tac-toe', frame)
        else:
            cv2.imshow('tic-tac-toe', frame)

        ch = 0xFF & cv2.waitKey(30)

        if ch == ord('q') or ch == 27:
            break

        if frame_number > frames_between_detection:
            frame_number = 0
        else:
            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
