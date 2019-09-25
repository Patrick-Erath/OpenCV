import numpy as numpy
import matplotlib.pyplot as plt
import cv2

# Sharpen the rice.png:

img_path = "./rice.png"
img = cv2.imread(img_path)

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape


# Created Blurred image using 7x7 Box, and make new copy
img_blur_box = cv2.blur(img_gray, (7,7))

# Create Blurred image using 7x7 Gaussian
img_blur_gaus = cv2.GaussianBlur(img_gray, (7,7), 0)

#1. Details (D) = original (I) - blurred (B)
# Iterate over row, col and replace each pixel
for row in range(0, height):
    for col in range(0, width):
        img_blur_box[row][col] = int(img_gray[row][col]) - int(img_blur_box[row][col])
        img_blur_gaus[row][col] = int(img_gray[row][col]) - int(img_blur_gaus[row][col])        

#2. Sharpened(S) = original (I) + details(D)
# Iterate over row, col and replace each pixel
for row in range(0, height):
    for col in range(0, width):
        img_blur_box[row][col] = int(img_gray[row][col]) + int(img_blur_box[row][col])
        img_blur_gaus[row][col] = int(img_gray[row][col]) + int(img_blur_gaus[row][col])
        
plt.subplot(131), plt.imshow(img_gray, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_blur_box, cmap='gray')
plt.title("Sharpening Using 7x7 Box"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_blur_gaus, cmap='gray')
plt.title("Sharpening Using 7x7 Gaus"), plt.xticks([]), plt.yticks([])
plt.show()