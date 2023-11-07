import cv2 as cv 
import numpy as np

# Create a simple binary image for illustration
image = cv.imread('noise.png', 0)


result = cv.erode(image, cv.getStructuringElement(cv.MORPH_CROSS, (3, 6)), iterations=1)
cv.imshow('E1', result)
result = cv.erode(result, cv.getStructuringElement(cv.MORPH_RECT, (1, 2)), iterations=2)
cv.imshow('E2', result)
result = cv.dilate(result, cv.getStructuringElement(cv.MORPH_CROSS, (1, 2)), iterations=1)
cv.imshow('D1', result)
result = cv.morphologyEx(result, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (4, 7)))
cv.imshow('D2', result)
cv.waitKey(0)
cv.destroyAllWindows()