from edge_detection import edge_detection
from edge_detection import canny_edge
import cv2
import matplotlib.pyplot as plt
from Preprocess_target import extract_red
from Preprocess_target import convert_to_gray
import numpy as np


def create_template(file):
    """
    this function creates the template form the
    template image
    :param file:
    :return:
    """
    img = cv2.imread(file, 1)
    grey_img = convert_to_gray(img)
    edge_map = canny_edge(grey_img, sigma=1, kernel_size=5, weak=100, strong=200, ltr=0.5, htr=0.6)
    edge_map = np.uint8(edge_map)
    # edge_map = cv2.Canny(grey_copy, 100, 200)
    cv2.imshow("result", edge_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./template/edge_template.PNG", edge_map)


if __name__ == '__main__':
    create_template("./template/EXIT_original_template.PNG")
