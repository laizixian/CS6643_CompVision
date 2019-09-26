import numpy as np
import cv2
import sys


def convolution(mask, img, new_img, boundary, axis):
    """
    this function calculates the convolution of img * mask[x]
    :param mask:
    :param img: the old image
    :param new_img: the processed image that needs to be returned
    :param boundary: the boundary of the unfiltered image
    :param axis: tells the function which axis the filter is applied on
                 0 for both x and y
                 1 for y
                 2 for x
    :return: new image after convolution
    """
    if axis is 1:
        for x in range(0 + boundary, img.shape[0] - boundary):
            for y in range(0, img.shape[1]):

                pixel = np.sum(np.multiply(img[x - boundary:x + boundary + 1,
                                           y],
                                           mask))
                new_img[x][y] = pixel
        return new_img

    elif axis is 2:
        for x in range(0, img.shape[0]):
            for y in range(0 + boundary, img.shape[1] - boundary):

                pixel = np.sum(np.multiply(img[x,
                                           y - boundary:y + boundary + 1],
                                           mask))
                new_img[x][y] = pixel
        return new_img

    else:
        for x in range(0 + boundary, img.shape[0] - boundary):
            for y in range(0 + boundary, img.shape[1] - boundary):

                pixel = np.sum(np.multiply(img[x - boundary:x + boundary + 1,
                                           y - boundary:y + boundary + 1],
                                           mask))
                new_img[x][y] = pixel
        return new_img


def gaussian_filter(g_filter, img):
    """
    This function apply the gaussian filter as separable filter on the image
    :param g_filter: a 1D gaussian filter mask
    :param img: a numpy array of the image
    :return: none
    """
    # calculate the boundary value
    boundary = int(g_filter.shape[0] / 2)

    # copy a intermediate image for separation and set data type to float
    imm_img = np.copy(img)
    imm_img = imm_img.astype(float)

    # copy a new image for storing and set data type to float
    new_img = np.copy(img)
    new_img = new_img.astype(float)

    # convolution the image with the y
    imm_img = convolution(g_filter, img, imm_img, boundary, 1)

    # convolution the image with the filter x
    new_img = convolution(g_filter, imm_img, new_img, boundary, 2)
    return new_img


def sobel_filter(img):
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], np.float32)
    ky = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], np.float32)
    length = img.shape[0]
    width = img.shape[1]
    Ix = np.zeros([img.shape[0], img.shape[1]])
    Iy = np.zeros([img.shape[0], img.shape[1]])

    for i in range(1, length-1):
        for j in range(1, width-1):
            Ix[i][j] = np.sum(np.multiply(img[i - 1:i + 2, j - 1:j + 2], kx))
            Iy[i][j] = np.sum(np.multiply(img[i - 1:i + 2, j - 1:j + 2], ky))

    G = np.hypot(Ix, Iy)
    G = G/G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression(img, theta):
    length = img.shape[0]
    width = img.shape[1]
    img1 = np.zeros([length, width])
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180.0

    for i in range(1, length-1):
        for j in range(1, width-1):
            temp1 = temp2 = 255
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                temp1 = img[i][j+1]
                temp2 = img[i][j-1]
            elif (22.5 <= angle[i][j] <67.5):
                temp1 = img[i+1][j-1]
                temp2 = img[i-1][j+1]
            elif (67.5 <= angle[i][j] < 112.5):
                temp1 = img[i+1][j]
                temp2 = img[i-1][j]
            elif (112.5 <= angle[i][j] < 157.5):
                temp1 = img[i-1][j-1]
                temp2 = img[i+1][j+1]

            if (img[i][j] >= temp1) and (img[i][j] >= temp2):
                img1[i][j] = img[i][j]
            else:
                img1[i][j] = 0

    return img1


def double_threshold(img, lowThresholdRatio, highThresholdRatio, weak, strong):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    length = img.shape[0]
    width = img.shape[1]
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    strong_i, strong_j = np.where(img >= highThreshold)
    img1 = np.zeros([length, width], dtype=np.int32)
    img1[strong_i, strong_j] = strong
    img1[weak_i, weak_j] = weak

    return img1


def hysteresis(img, weak, strong = 255):
    length = img.shape[0]
    width = img.shape[1]
    for i in range(1, length-1):
        for j in range(1, width-1):
            if img[i][j] == weak:
                for i1 in range(i-1, i+2):
                    for j1 in range(j-1, j+2):
                        if (i1 == i and j1 == j):
                            continue
                        if img[i1][j1] == strong:
                            img[i][j] = strong
                        else:
                            img[i][j] = 0
    return img


def thresholding(threshold, data):
    """
    This function thresholds the image data and return
    the thresholded image data
    :param threshold:
    :param data:
    :return: new_data
    """
    new_data = np.copy(data)
    new_data[new_data > threshold] = 255
    new_data[new_data < threshold] = 0
    return new_data


def thinning(size, img):
    """
    This function produce a thin edge map by marking
    zero crossing
    :param img: the image array that needs process
    :param size: the size of the filter
    :return new_img: the new edge map after thinning
    """
    new_img = np.copy(img)
    x_boundary = int(size[0] / 2)
    y_boundary = int(size[1] / 2)
    for x in range(0 + x_boundary, img.shape[0] - x_boundary):
        for y in range(0 + y_boundary, img.shape[1] - y_boundary):
            if img[x][y] == 255:
                total = np.sum(img[x - x_boundary:x + x_boundary + 1,
                               y - y_boundary:y + y_boundary + 1])
                if total < size[0] *size[1] * 255:
                    new_img[x][y] = 255
                else:
                    new_img[x][y] = 0

    return new_img


def normalization(array_min, array_max, target_min, target_max, data):
    """
    This function normalize the data array to the desired array
    :param array_min: the min value of the array
    :param array_max: the max value of the array
    :param target_min: the min value of he target range
    :param target_max: the max value of the target range
    :param data: the array data that needs normalization
    :return: the normalized data
    """
    normalized = np.subtract(data, array_min)
    normalized = np.divide(normalized, np.subtract(array_max, array_min))
    normalized = np.multiply(normalized, target_max - target_min)
    return normalized


def generate_gaussian(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def edge_detection(s_img, sigma):

    g_filter = generate_gaussian(5, sigma)
    gaussian_filtered = gaussian_filter(g_filter, s_img)
    l_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplace_filtered = np.copy(gaussian_filtered)
    laplace_filtered = convolution(l_filter, gaussian_filtered, laplace_filtered, 1, 0)
    laplace_filtered = laplace_filtered.astype(int)
    laplace_filtered_threshold = thresholding(0, laplace_filtered)
    crossing_size = (3, 3)
    thin_edge_map = thinning(crossing_size, laplace_filtered_threshold)

    return thin_edge_map


def canny_edge(img, sigma, kernel_size, weak, strong, ltr, htr):
    g_filter = generate_gaussian(kernel_size, sigma)
    gaussian_filtered = gaussian_filter(g_filter, img)
    sobel_filtered, theta = sobel_filter(gaussian_filtered)
    suppressed = non_max_suppression(sobel_filtered, theta)
    thresholded = double_threshold(suppressed, ltr, htr, weak, strong)
    thresholded = np.uint8(thresholded)
    thin_edge = hysteresis(thresholded, weak, strong)
    return thin_edge
