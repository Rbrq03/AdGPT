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


def check_args(args):

    assert (
        args.version == "en" or args.version == "cn"
    ), "Currently AdGPT only support cn & en version, version argument shoule be one of them"
    assert args.openai or args.chatglm, "Chat model shoule be one of openai or chatglm"
    assert args.openai == False or (
        args.openai_key is not None and args.openai_base is not None
    ), "Openai mdoel shoule have api key and api base"
    assert args.chatglm == False or (
        args.glm_model is not None and args.glm_key is not None
    )
