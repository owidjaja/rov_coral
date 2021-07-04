import cv2

def auto_resize(img, target_width=800):
    # print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print("Resized Dimension: ", resized.shape, '\n')
    return resized

old = cv2.imread("./sample/coral_reference.jpg")
if old is None:
    exit("failed to read image")

cv2.imshow("old", old)

resized = auto_resize(old, target_width=1200)
cv2.imshow("resized", resized)

cv2.imwrite("./sample/coral_ref.jpg", resized)

cv2.waitKey(0)