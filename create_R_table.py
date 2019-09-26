import cv2
import numpy as np
from collections import defaultdict


def calculate_center(img):
    """
    pick the center point for the four letter template img
    :param img: the image of the template
    :return the center for the four letter:
    """
    xy = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    return xy


def calculate_gradient(img):
    """
    get the gradient mapping og the image
    :param img: the source image
    :return gradient: the gradient map
    """
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    length = img.shape[0]
    width = img.shape[1]
    Ix = np.zeros([img.shape[0], img.shape[1]])
    Iy = np.zeros([img.shape[0], img.shape[1]])

    for x in range(1, length - 1):
        for y in range(1, width - 1):
            Ix[x][y] = np.sum(np.multiply(img[x - 1:x + 2, y - 1:y + 2], kx))
            Iy[x][y] = np.sum(np.multiply(img[x - 1:x + 2, y - 1:y + 2], ky))

    gradient = np.arctan2(Iy, Ix)
    return gradient


def calculate_r_table(path):
    """
    calculate the r_table for the image
    :param path: te path for the template image
    :return: r_table
    """
    img = cv2.imread(path, 0)
    xy = calculate_center(img)
    gradient_map = calculate_gradient(img)
    r_table = defaultdict(list)
    have_edge = np.transpose(np.nonzero(gradient_map))
    for x, y in have_edge:
        r = (xy[0] - x, xy[1] - y)
        r_table[gradient_map[x, y]].append(r)

    return r_table


if __name__ == '__main__':
    print("find the R table of the 4 object")
    calculate_r_table("./template/edge_template_E.png")
