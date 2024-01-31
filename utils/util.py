import cv2
import numpy
from PIL import Image


def cv2pil(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def pil2cv(image):
    image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def crop_img(img, boxes):
    a = int(boxes[0])
    b = int(boxes[2])
    c = int(boxes[1])
    d = int(boxes[3])
    crop = img[c:d, a:b, :]
    return crop


def area(boxes):
    a = int(boxes[0])
    b = int(boxes[2])
    c = int(boxes[1])
    d = int(boxes[3])
    return (b - a) * (d - c)