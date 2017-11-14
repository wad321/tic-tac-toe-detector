import cv2
import glob
import imutils


def crop_middle_square(image):
    if image.shape[0] != image.shape[1]:
        w = image.shape[0]
        h = image.shape[1]
        if w > h:
            image = image[int((w-h)/2):int((w+h)/2), 0:h].copy()
        else:
            image = image[0:w, int((h-w)/2):int((h+w)/2)].copy()
    return image


def change_image_size(image, width):
    image = imutils.resize(image, width=width)
    return image


def make_em_smaller(globfilepath):
    for file in glob.glob(globfilepath):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        cropped = crop_middle_square(image)
        sized = change_image_size(cropped, new_width)
        cv2.imwrite(file, sized)


if __name__ == '__main__':
    new_width = 64
    make_em_smaller("kolka/*.jpg")
    make_em_smaller("krzyzyki/*.jpg")
    make_em_smaller("puste/*.jpg")
