import sys
import cv2
from Preprocess_target import extract_red
from Preprocess_target import convert_to_gray
import numpy as np
from edge_detection import canny_edge
from create_R_table import calculate_r_table
from find_object_location import mark_location
import copy

if __name__ == '__main__':
    video_name = sys.argv[1]

    cap = cv2.VideoCapture("./test_video/" + video_name)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('./video_result/result_1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    path_E = "./template/edge_template_E.png"
    path_X = "./template/edge_template_X.png"

    char_E = calculate_r_table(path_E)
    char_X = calculate_r_table(path_X)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    i = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print(i)
            i += 1
            image_copy = copy.copy(frame)
            red_img = extract_red(image_copy)
            grey_img = convert_to_gray(red_img)
            edge_map = canny_edge(grey_img, sigma=1, kernel_size=5, weak=50, strong=255, ltr=0.05, htr=0.16)
            edge_map = np.uint8(edge_map)
            location_e, matrix_e, max_value_e = mark_location(char_E, edge_map, 1, 1)
            location_x, matrix_x, max_value_x = mark_location(char_X, edge_map, 1, 1)

            max_value = [max_value_e, max_value_x]
            scale = [location_e[2], location_x[2]]
            pick_scale = scale[max_value.index(max(max_value))]
            dummy_e = matrix_e[:, :, pick_scale]
            location_e = np.where(dummy_e == dummy_e.max())
            print(location_e)

            dummy_x = matrix_x[:, :, pick_scale]
            location_x = np.where(dummy_x == dummy_x.max())
            print(location_x)

            dummy_e = np.uint8(dummy_e)
            dummy_x = np.uint8(dummy_x)

            cv2.circle(frame, (location_x[1] + int((location_x[1] - location_e[1]) / 2),
                               int((location_e[0] + location_x[0]) / 2)), 5, (0, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(pick_scale / 20), (location_x[1] + int((location_x[1] - location_e[1]) / 2),
                                                      int((location_e[0] + location_x[0]) / 2)), font, 1, (0, 255, 255),
                        2, cv2.LINE_AA)
            out.write(frame)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

