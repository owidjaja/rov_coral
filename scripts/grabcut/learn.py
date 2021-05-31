import cv2
import numpy as np

# black_img = np.zeros((800,800), dtype=np.uint8)
# cv2.rectangle(black_img, (200,200), (600,600), [255,0,0], -1)
# cv2.imshow("black", black_img)
# print(black_img.shape)

# cv2.circle(black_img, (400, 400), radius=50, color=[0,0,255], thickness=-1)

# cv2.imshow("CIRCLE", black_img)
# cv2.waitKey(0)

# Python program to explain cv2.circle() method



# image = cv2.imread("coral_mask.jpg")
# window_name = 'Image'
# print(image.shape)

# mask = np.zeros(image.shape[:2], dtype = np.uint8)
# print(mask.shape)

# center_coordinates = (300, 500)
# radius = 20
# color = (255, 0, 0)
# thickness = -1
# image = cv2.circle(image, center_coordinates, radius, color, thickness)

# cv2.imshow(window_name, image)
# cv2.waitKey(0)

# height, width, depth = image.shape

# for i in range(0, height):      # y
#     for j in range(0, width):   # x
#         if np.any(image[i, j] == 0):
#             cv2.circle(image, (j-1, i-1), 1, color, thickness)
#         else:
#             # print(i, j, "WTF")
#             cv2.circle(image, (j-1, i-1), 1, (0,255,0), thickness)

# cv2.imshow("AFTER OUR THING", image)
# cv2.waitKey(0)


a = np.zeros((300, 300, 3), dtype= np.uint8) # Original image. Here we use a black image but you can replace it with your video frame.                                                                                                        
cv2.rectangle(a, (100,100), (200,200), [0,255,0], -1)

white = np.full((300, 300, 3), 255, dtype=np.uint8)  # White image  

cv2.imshow("orig image", a)
cv2.imshow("white image", white)

found = np.where(a[:,:] == [0,0,0])   # This is where we replace black pixels.
print(found)

# get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
black_pixels = np.where(
    (a[:, :, 0] == 0) & 
    (a[:, :, 1] == 0) & 
    (a[:, :, 2] == 0)
)
print(black_pixels)

# set those pixels to white
a[found] = [255, 0, 0]

cv2.imshow("result image", a)
cv2.waitKey(0)