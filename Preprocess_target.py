import cv2
from edge_detection import edge_detection
import matplotlib.pyplot as plt
import numpy as np


def convert_to_gray(img):
    """
    convert the BRG image data to gray image data
    :param img: BGR image data as numpy array
    :return: gray scale image data as numpy array
    """
    # split pixel into different b g r colors
    b, g, r = cv2.split(img)
    # calculate the gray color
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return gray


def extract_red(img):
    """
    this function extract red color from the
    image
    :param img: source image
    :return red_img: the red image of the source image
    """
    red_img = np.zeros(img.shape)
    logic = np.zeros((img.shape[0], img.shape[1]))
    np.logical_and(img[:, :, 2] > 50, img[:, :, 1] < 100, logic)
    np.logical_and(img[:, :, 0] < 100, logic, logic)

    red_img[logic.astype(bool)] = [0, 0, 255]
    return red_img

