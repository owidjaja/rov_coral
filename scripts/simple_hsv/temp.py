import cv2

src = cv2.imread("../res/coral_night.jpg")
mask = cv2.imread("coral_mask.jpg", 0)

_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
print(mask.shape)

cv2.imwrite("masked_perfect_coral.jpg", cv2.bitwise_and(src,src,mask=thresh))