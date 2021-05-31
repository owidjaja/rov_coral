import cv2
import numpy as np
import matplotlib.pyplot as plt

def cb_nothing(param):
    pass




if __name__ == "__main__":
    image_original = cv2.imread('../res/coral_under3.jpg', cv2.IMREAD_COLOR)
    cv2.imshow("Input", image_original)

    # trackbar
    cv2.namedWindow("Trackbar_Window")
    cv2.createTrackbar("thresh1", "Trackbar_Window", 30, 180, cb_nothing)
    cv2.createTrackbar("thresh2", "Trackbar_Window", 50, 255, cb_nothing)

    while (True):
        cv2.getTrackbarPos("thresh1", "Trackbar_Window")
        cv2.getTrackbarPos("thresh2", "Trackbar_Window")

        # remove noise
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.Canny(image_gray, threshold1=20, threshold2=200)
        cv2.imshow("Output", filtered_image)


    # Plot outputs
    # (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 15))
    # ax1.title.set_text('Original Image')
    # ax1.imshow(image_original)

    # ax2.title.set_text('Canny Image')
    # ax2.imshow(filtered_image, cmap='gray')

    # plt.show()