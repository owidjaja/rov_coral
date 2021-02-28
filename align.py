import cv2
import numpy as np

def remove_noise(img, ksize=5, thresh_val=32):
    """ Remove noise from img using medianBlur """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, ksize)
    # cv2.imshow("blurred", blur)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # https://stackoverflow.com/questions/42920810/in-opencv-is-mask-a-bitwise-and-operation
    res = cv2.bitwise_and(img, img, mask=thresh)
    return res

def closing(img, ksize=5):
    kernel = np.ones((ksize,ksize), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

def perspective_transformation(img, x, y, w, h):
    # https://www.youtube.com/watch?v=mzhiKpM8eJ0
    # https://www.youtube.com/watch?v=j4el1XARYSo

    rows, cols, ch = img.shape

    coor_orig_img = np.float32([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])

    # cv2.imshow("Input", img)

    height, width = h, w
    coor_new_img = np.float32([[0,0], [width,0], [0,height], [width,height]])

    # get perspective transformation matrix
    matrix_perspective = cv2.getPerspectiveTransform(coor_orig_img, coor_new_img)   

    # perform transformation
    perspective = cv2.warpPerspective(img, matrix_perspective, (width, height))

    # cv2.imshow("Output", perspective)
    cv2.imshow("pers", perspective)

def enlarge_image(original_image, factor=2):
    original_height, original_width = original_image.shape[:2]
    resized_image = cv2.resize(original_image, (int(original_height*factor), int(original_width*factor)), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('resized_image.jpg',resized_image)
    return resized_image


IMAGES = ["NewMask.jpg", "NewMask_under.jpg", "front_flip.jpg", "OldMask_Flip.jpg"]
old = cv2.imread(IMAGES[3], cv2.IMREAD_GRAYSCALE)
new = cv2.imread(IMAGES[1], cv2.IMREAD_GRAYSCALE)

cv2.imshow("ORIGINAL_new", new)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(len(contours))

biggest_contour = contours[0]
max_area = cv2.contourArea(biggest_contour)
for cont in contours:
    this_area = cv2.contourArea(cont)
    if this_area > max_area:
        biggest_contour = cont
        max_area = this_area

x, y, w, h = cv2.boundingRect(biggest_contour)

# To give some extension on window
EXTEND_DIMENSION = 0.1
x0 = int(x - (EXTEND_DIMENSION * w))
y0 = int(y - (EXTEND_DIMENSION * h))
x1 = int(x + ((1+EXTEND_DIMENSION) * w))
y1 = int(y + ((1+EXTEND_DIMENSION) * h))

# if x0 < 0:
#     x0 = 0
# if y0 < 0:
#     y0 = 0
# # TODO: have not tested
# if x1 > new.shape[1]:
#     x1 = new.shape[1]
# if y1 > new.shape[0]:
#     y1 = new.shape[0]

print(x,y,x+w,y+h)
print(x0,y0,x1,y1)

coordinates_cont = np.float32([[x0,y0], [x1,y0], [x0,y1], [x1,y1]])

height, width = h, w
coordinates_new_img = np.float32([[0,0], [width,0], [0,height], [width,height]])

matrix_perspective = cv2.getPerspectiveTransform(coordinates_cont, coordinates_new_img)
perspective = cv2.warpPerspective(new, matrix_perspective, (width, height))
cv2.imshow("new", new)
cv2.imshow("pers", perspective)

TARGET_DIM = 600
big_pers = enlarge_image(perspective, TARGET_DIM/perspective.shape[0])

# canvas_shape = [new.shape[0], new.shape[1], 3]
# canvas = np.zeros(canvas_shape, dtype=np.uint8)
# cv2.drawContours(canvas, [biggest_contour], -1, [0,255,0], thickness=-1)
# cv2.imshow("canvas", canvas)

# canvas_bin = np.zeros(new.shape, dtype=np.uint8)
# cv2.drawContours(canvas_bin, [biggest_contour], -1, 255, thickness=1)
# cv2.imshow("canvas_binary", canvas_bin)

# blurred = cv2.medianBlur(canvas_bin, 3)
# cv2.imshow("new_blurred", blurred)

print(big_pers.shape)
ret, thresh = cv2.threshold(big_pers, 127, 255, cv2.THRESH_BINARY)
closed = closing(thresh, ksize=9)
print(closed.shape)
cv2.imshow("new_closed", closed)

big_old  = enlarge_image(old, TARGET_DIM/old.shape[0])
old_closed = closing(big_old, 9)
cv2.imshow("old", old_closed)

cv2.waitKey(0)
cv2.destroyAllWindows()