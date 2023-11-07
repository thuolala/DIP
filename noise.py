import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np
import random 

# -------------------------------------- #
# TASK 2 
# Read input img as grayscale
input2 = cv.imread("input2.png", cv.IMREAD_GRAYSCALE)  
height, width = input2.shape


# ----------------------Extract only area with a lot of noise then pre-processing it----------------------
# Blur
blur_noise = cv.medianBlur(input2, 7) 

# Threshold
th1 = cv.adaptiveThreshold(blur_noise, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
ret, th2 = cv.threshold(th1, 0, 255, cv.THRESH_BINARY_INV)

img_lotNoise = th2[270:height, 260:width]  

# Pre-processing that region 
# Opening in vertical
kernel_noise1 = cv.getStructuringElement(cv.MORPH_RECT, (6, 1))
opening_noise1 = cv.morphologyEx(img_lotNoise, cv.MORPH_OPEN, kernel_noise1)

# Opening in horizonal
kernel_noise2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 6))
opening_noise2 = cv.morphologyEx(opening_noise1, cv.MORPH_OPEN, kernel_noise2)

# Opening in cross 
kernel_noise3 = cv.getStructuringElement(cv.MORPH_CROSS, (1, 6))
opening_noise3 = cv.morphologyEx(opening_noise2, cv.MORPH_OPEN, kernel_noise3)

# Closing 
kernel_noise4 = np.ones((4, 3),np.uint8)
closing_noise = cv.morphologyEx(opening_noise3, cv.MORPH_CLOSE, kernel_noise4)

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
ret, closing_noise = cv.threshold(closing_noise, 0, 255, cv.THRESH_BINARY_INV)
opening[270:height, 260:width] = closing_noise

# Display it
cv.imshow("After Cleaning Up", opening)

converted_img = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)

opening = cv.fastNlMeansDenoisingColored(converted_img,None,50,50,10,50)
cv.imshow("After Denosing", opening)

# # ---------------------- Draw contours----------------------
# # Contour 
# img = cv.imread("input2.png")
# ret, thresh = cv.threshold(opening, 0, 255, cv.THRESH_BINARY_INV)
# # Detect contours
# contours, hie = cv.findContours(image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE) 
# # Draw contours
# img_copy = img.copy() 

# # Get sum of h & w 
# count = 0  
# sum_h = 0 
# sum_w = 0 
# for cnt in contours:
#     x,y,w,h = cv.boundingRect(cnt) 
#     sum_h += h 
#     sum_w += w 
#     count += 1 

# # Get mean of h & w 
# mean_h = sum_h / count 
# mean_w = sum_w / count 

# for cnt in contours:
#     # Random color 
#     R = random.randint(0, 255)
#     G = random.randint(0, 255)
#     B = random.randint(0, 255)
#     x,y,w,h = cv.boundingRect(cnt)
#     if (h >= mean_h):
#         img_copy = cv.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
# cv.imwrite("output2.png", img_copy)
# cv.imshow("Drawed Contours Image", img_copy)

cv.waitKey(0)
cv.destroyAllWindows()