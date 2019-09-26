import cv2

if __name__ == '__main__':
    img = cv2.imread("./template/EXIT_original_template.PNG")
    cv2.imshow("original", img)
    new_img = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow("scaled", new_img)
    cv2.imwrite("./template/EXIT_original_template_0.3.PNG", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()