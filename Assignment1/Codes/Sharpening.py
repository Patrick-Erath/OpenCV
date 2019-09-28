import cv2
import matplotlib.pyplot as plt 
import numpy as np
# Sharpen the rice.png:

img_path = "../rice.png"
img = cv2.imread(img_path)

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape

# Created Blurred image using 7x7 Box, and make new copy
img_blur_box = cv2.blur(img_gray, (9,9))

# Create Blurred image using 7x7 Gaussian
img_blur_gaus = cv2.GaussianBlur(img_gray, (9,9), 0)

#1. Details (D) = original (I) - blurred (B)
# Replace each pixel
details_box = img_gray - img_blur_box
details_gaus = img_gray - img_blur_gaus     

#2. Sharpened(S) = original (I) + details(D)
# Replace each pixel
sharp_box = img_gray + details_box
sharp_gaus = img_gray + details_gaus
 
# Display
plt.subplot(231), plt.imshow(img_blur_box, cmap='gray')
plt.title("Box Blurred"), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(details_box, cmap='gray')
plt.title("Details-Box"), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(sharp_box, cmap='gray')
plt.title("Sharpened-Box"), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(img_blur_gaus, cmap='gray')
plt.title("Gaussian Blurred"), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(details_gaus, cmap='gray')
plt.title("Details-Gaussian"), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(sharp_gaus, cmap='gray')
plt.title("Sharpened-Gaussian"), plt.xticks([]), plt.yticks([])
plt.show()