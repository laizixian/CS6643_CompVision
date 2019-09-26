import cv2
import numpy as np
from create_R_table import calculate_gradient
from create_R_table import calculate_r_table
from Preprocess_target import extract_red
from Preprocess_target import convert_to_gray
import copy
from edge_detection import canny_edge
from edge_detection import edge_detection
from multiprocessing import Pool
import sys


def accumulate_gradient(r_table, edge_img, scale, rotation):
    """
    accumulate the gradient by compare r table with
    the edge image of the source image
    :param r_table:
    :param edge_img:
    :return : accumulate_matrix
    """
    x_size = edge_img.shape[0]
    y_size = edge_img.shape[1]
    max_value = 0
    max_location = (0, 0)
    gradient_map = calculate_gradient(edge_img)
    has_gradient = np.transpose(np.nonzero(gradient_map))

    # create scale
    scale_line = np.linspace(0, scale, scale*20 + 1)
    #accumulate_matrix = np.zeros([x_size, y_size, len(scale_line), rotation * 2], dtype=int)
    accumulate_matrix = np.zeros([x_size, y_size, len(scale_line)], dtype=int)
    for x, y in has_gradient:
        print(x, y)
        for s in scale_line:
            if gradient_map[x, y] in r_table.keys():
                for r in r_table[gradient_map[x, y]]:
                    xy = (int(x + ((r[0]) * s)), (int(y + ((r[1]) * s))))
                    if 0 < xy[0] < x_size and 0 < xy[1] < y_size:
                        accumulate_matrix[xy[0], xy[1], int(s * 20)] += 1
                        if accumulate_matrix[xy[0], xy[1], int(s * 20)] >= max_value:
                            max_value = accumulate_matrix[xy[0], xy[1], int(s * 20)]
                            max_location = (xy[0], xy[1], int(s * 20))

    return max_location, accumulate_matrix, max_value


def mark_location(char, edge_map, scale, rotation):
    acc_char, matrix_char, max_value = accumulate_gradient(char, edge_map, scale, rotation)
    return acc_char, matrix_char, max_value


if __name__ == '__main__':

    # load the target image
    # test_file = "test3_3.jpg"
    test_file = sys.argv[1]
    red_flag = sys.argv[2]
    image = cv2.imread("./test_img/" + test_file, 1)
    image_copy = copy.copy(image)

    if red_flag == "1":
        red_img = extract_red(image_copy)

        cv2.imwrite("./Red_image/red_" + test_file, red_img)
    else:
        red_img = image_copy

    grey_img = convert_to_gray(red_img)

    cv2.imwrite("./Grey_img/grey_" + test_file, grey_img)

    edge_map = canny_edge(grey_img, sigma=1, kernel_size=5, weak=50, strong=255, ltr=0.05, htr=0.16)
    edge_map = np.uint8(edge_map)
    cv2.imwrite("./edge_result/edge_" + test_file, edge_map)


    path_E = "./template/edge_template_E.png"
    path_X = "./template/edge_template_X.png"

    char_E = calculate_r_table(path_E)
    char_X = calculate_r_table(path_X)

    location_e, matrix_e, max_value_e = mark_location(char_E, edge_map, 1, 1)
    location_x, matrix_x, max_value_x = mark_location(char_X, edge_map, 1, 1)

    max_value = [max_value_e, max_value_x]
    scale = [location_e[2], location_x[2]]
    pick_scale = scale[max_value.index(max(max_value))]
    dummy_e = matrix_e[:, :, pick_scale]
    location_e = np.where(dummy_e == dummy_e.max())

    dummy_x = matrix_x[:, :, pick_scale]
    location_x = np.where(dummy_x == dummy_x.max())

    dummy_e = np.uint8(dummy_e)
    dummy_x = np.uint8(dummy_x)

    cv2.imshow("dummy_e", dummy_e)
    cv2.imwrite("./Contour_img/E_contour_" + test_file, dummy_e)

    cv2.imshow("dummy_x", dummy_x)
    cv2.imwrite("./Contour_img/X_contour_" + test_file, dummy_x)

    cv2.circle(image, (location_x[1] + int((location_x[1] - location_e[1]) / 2),
                       int((location_e[0] + location_x[0]) / 2)), 5, (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(pick_scale / 20), (location_x[1] + int((location_x[1] - location_e[1]) / 2),
                                              int((location_e[0] + location_x[0]) / 2)), font, 1, (0, 255, 255),
                2, cv2.LINE_AA)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./result/result_" + test_file, image)


