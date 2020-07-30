import cv2
import numpy as np

cam = cv2.VideoCapture(0)
lower_HSV_values1 = np.array([0, 40, 0], dtype = "uint8")
upper_HSV_values1 = np.array([25, 255, 255], dtype = "uint8")

lower_HSV_values2 = np.array([0, 100, 100], dtype = "uint8")
upper_HSV_values2 = np.array([0, 255, 255], dtype = "uint8")

lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

while(cam.isOpened()):
    ret,img=cam.read()
    img = np.flip(img, axis=1)
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(img ,cv2.COLOR_BGR2YCR_CB)

    mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV1 = cv2.inRange(HSV_image, lower_HSV_values1, upper_HSV_values1)
    mask_HSV2 = cv2.inRange(HSV_image, lower_HSV_values2, upper_HSV_values2)

    binary_mask_image = cv2.add(mask_HSV1,mask_HSV2, mask_YCbCr)
    image_foreground = cv2.erode(binary_mask_image, None, iterations=3)
    dilated_binary_image = cv2.dilate(binary_mask_image, None,iterations=3)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

    image_marker = cv2.add(image_foreground,image_background)
    image_marker32 = np.int32(image_marker)

    cv2.watershed(img, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)

    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output = cv2.bitwise_and(img, img, mask=image_mask)

    cv2.imshow("images",output)
    k = cv2.waitKey(10)
    if k == 27:
        break
