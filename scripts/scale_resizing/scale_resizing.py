import cv2

def scale_resizing(img, scale):
    print("Original Dimension: ", img.shape)

    width = int(img.shape[1] * (scale/100))
    height= int(img.shape[0] * (scale/100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print("Resized Dimension: ", resized.shape)
    return resized