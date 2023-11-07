import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np
import random 

# # TASK 1

# # Read input img 
# input1 = cv.imread("input1.jpg", cv.IMREAD_COLOR) 
# # Display input img 
# cv.imshow('Original', input1)

# # 1a)
# # Yellow 
# yellow = np.uint8([[[255,255,0]]])
# hsv_yellow = cv.cvtColor(yellow,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_yellow = np.array([27,50,50])
# upper_yellow = np.array([30,255,255])
# mask = cv.inRange(hsv, lower_yellow, upper_yellow)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Yellow Star',res)
# cv.imwrite("yellow_star.png", res)

# # Orange 
# orange = np.uint8([[[255,165,0]]])
# hsv_orange = cv.cvtColor(orange,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_orange = np.array([0,150,20])
# upper_orange = np.array([13,255,255])
# mask = cv.inRange(hsv, lower_orange, upper_orange)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Orange Star',res)
# cv.imwrite("orange_star.png", res)

# # Pink 
# pink = np.uint8([[[255,192,203]]])
# hsv_pink = cv.cvtColor(pink,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_pink = np.array([159,245,183])
# upper_pink = np.array([179,265,263])
# mask = cv.inRange(hsv, lower_pink, upper_pink)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Pink Star',res)
# cv.imwrite("pink_star.png", res)

# # Blue 
# blue = np.uint8([[[0,0,255]]])
# hsv_blue = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_blue = np.array([92,245,153])
# upper_blue = np.array([112,265,233])
# mask = cv.inRange(hsv, lower_blue, upper_blue)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Blue Star',res)
# cv.imwrite("blue_star.png", res)

# # Green 
# green = np.uint8([[[0,255,0]]])
# hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_green = np.array([50,50,50])
# upper_green = np.array([80,255,255])
# mask = cv.inRange(hsv, lower_green, upper_green)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Green Star',res)
# cv.imwrite("green_star.png", res)

# # Purple 
# purple = np.uint8([[[255,0,255]]])
# hsv_purple = cv.cvtColor(purple,cv.COLOR_BGR2HSV)
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_purple = np.array([116,179,83])
# upper_purple = np.array([166,255,255])
# mask = cv.inRange(hsv, lower_purple, upper_purple)
# res = cv.bitwise_and(input1, input1, mask = mask)
# cv.imshow('Purple Star',res)
# cv.imwrite("purple_star.png", res)


# # 1b) 
# # Read input img as grayscale
# input1_gray = cv.imread("input1.jpg", cv.IMREAD_GRAYSCALE) 
# # cv.imshow('Grayscale', input1_gray)

# # Change white borders to black borders 
# th, dst = cv.threshold(input1_gray, 220, 0, cv.THRESH_TOZERO_INV)
# cv.imshow('Black borders', dst)
# cv.imwrite("black_border.png", dst)

# # 1c) 
# # Create mask for gray  
# hsv = cv.cvtColor(input1, cv.COLOR_BGR2HSV)
# lower_gray = np.array([8,95,176])
# upper_gray = np.array([18,115,256])
# mask = cv.inRange(hsv, lower_gray, upper_gray)

# # Clean up
# # Closing then Opening
# kernel = np.ones((5, 5),np.uint8)
# mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
# mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

# cv.imshow('All black', mask)
# cv.imwrite("all_black.png", mask)
# cv.waitKey()  
# cv.destroyAllWindows() 

# -------------------------------------- #
# TASK 2 
# Read input img as grayscale
input2 = cv.imread("input2.png", cv.IMREAD_GRAYSCALE)  
height, width = input2.shape


# ----------------------Extract only area with a lot of noise then pre-processing it----------------------
# Blur
blur_noise = cv.medianBlur(input2, 11) 

# Threshold
th1 = cv.adaptiveThreshold(blur_noise, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
ret, th2 = cv.threshold(th1, 0, 255, cv.THRESH_BINARY_INV)

img_lotNoise = th2[270:height, 260:width]  

cv.imwrite('noise.png', img_lotNoise)

# Pre-processing that area 
result = cv.erode(img_lotNoise, cv.getStructuringElement(cv.MORPH_CROSS, (3, 6)), iterations=1)
result = cv.erode(result, cv.getStructuringElement(cv.MORPH_RECT, (1, 2)), iterations=2)
result = cv.dilate(result, cv.getStructuringElement(cv.MORPH_CROSS, (1, 2)), iterations=1)
result = cv.morphologyEx(result, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (4, 7)))

# -----------------------------------------------------------------------------------------------------

# ----------------------------------Pre-processing the original image----------------------------------
# Blur
blur = cv.medianBlur(input2, 7) 

# Otsu threshold
thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# Opening
kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (4,3))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1)

# Re-attacht that region with original 
ret, result = cv.threshold(result, 0, 255, cv.THRESH_BINARY_INV)
opening[270:height, 260:width] = result

# Display it
cv.imshow("After Cleaning Up", opening)

# ---------------------- Draw contours----------------------
# Contour 
img = cv.imread("input2.png")
ret, thresh = cv.threshold(opening, 0, 255, cv.THRESH_BINARY_INV)
# Detect contours
contours, hie = cv.findContours(image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE) 
# Draw contours
img_copy = img.copy() 

# Get sum of h & w 
count = 0  
sum_h = 0 
sum_w = 0 
for cnt in contours:
    x,y,w,h = cv.boundingRect(cnt) 
    sum_h += h 
    sum_w += w 
    count += 1 

# Get mean of h & w 
mean_h = sum_h / count 
mean_w = sum_w / count 

for cnt in contours:
    # R = random.randint(0, 255)
    # G = random.randint(0, 255)
    # B = random.randint(0, 255)
    x,y,w,h = cv.boundingRect(cnt)
    if (h >= mean_h):
        img_copy = cv.rectangle(img_copy,(x,y),(x+w,y+h),(0, 255, 0),2)
cv.imwrite("output2.png", img_copy)
cv.imshow("Drawed Contours Image", img_copy)

cv.waitKey(0)
cv.destroyAllWindows()



# TEST 
# import cv2 as cv 
# import numpy as np

# # Create a simple binary image for illustration
# image = cv.imread('noise.png', 0)


# result = cv.erode(image, cv.getStructuringElement(cv.MORPH_CROSS, (7, 7)), iterations=1)
# cv.imshow('E1', result)
# result = cv.erode(result, cv.getStructuringElement(cv.MORPH_RECT, (3, 2)), iterations=1)
# cv.imshow('E2', result)
# result = cv.dilate(result, cv.getStructuringElement(cv.MORPH_CROSS, (2, 3)), iterations=1)
# cv.imshow('D1', result)
# result = cv.morphologyEx(result, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))
# cv.imshow('D2', result)
# cv.waitKey(0)
# cv.destroyAllWindows()