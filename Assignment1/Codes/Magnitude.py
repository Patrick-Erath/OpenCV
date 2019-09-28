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

kernel = np.ones((5, 5), np.float32)/(5*5)
# Applying intensity channels
I_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
I_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
I_m = cv2.magnitude(I_x, I_y)

I_x_sharp = cv2.Sobel(sharp_box, cv2.CV_64F, 1, 0, ksize=5)
I_y_sharp = cv2.Sobel(sharp_gaus, cv2.CV_64F, 0, 1, ksize=5)

I_x_box = cv2.Sobel(sharp_box, cv2.CV_64F, 1, 0, ksize=5)
I_y_box = cv2.Sobel(sharp_box, cv2.CV_64F, 0, 1, ksize=5)
I_m_box = cv2.magnitude(I_x, I_y)

I_x_gaus = cv2.Sobel(sharp_gaus, cv2.CV_64F, 1, 0, ksize=5)
I_y_gaus = cv2.Sobel(sharp_gaus, cv2.CV_64F, 0, 1, ksize=5)
I_m_gaus = cv2.magnitude(I_x_gaus, I_y_gaus)

plt.figure(figsize=(10, 3))
plt.subplot(131), plt.imshow(I_x_box, cmap='gray')
plt.title('X-axis Edge detection (Box)'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(I_y_box, cmap='gray')
plt.title('Y-axis Edge detection (Box)'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(I_m_box, cmap='gray')
plt.title("Magnitude (Box)"), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(10, 3))
plt.subplot(131), plt.imshow(I_x_gaus, cmap='gray')
plt.title('X-axis Edge detection (Gauss)'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(I_y_gaus, cmap='gray')
plt.title('Y-axis Edge detection (Gauss)'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(I_m_gaus, cmap='gray')
plt.title("Magnitude (Gauss)"), plt.xticks([]), plt.yticks([])

THRESH_1 = 220
THRESH_2 = 800

ret_1, img_1 = cv2.threshold(I_m, THRESH_1, 255, cv2.THRESH_BINARY)
ret_box_1, img_box_1 = cv2.threshold(I_m_box, THRESH_1, 255, cv2.THRESH_BINARY)
ret_gaus_1, img_gaus_1 = cv2.threshold(I_m_gaus, THRESH_2, 255, cv2.THRESH_BINARY)

ret_2, img_2 = cv2.threshold(I_m, THRESH_2, 255, cv2.THRESH_BINARY)
ret_box_2, img_box_2 = cv2.threshold(I_m_box, THRESH_2, 255, cv2.THRESH_BINARY)
ret_gaus_2, img_gaus_2 = cv2.threshold(I_m_gaus, THRESH_2, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10,3))
plt.subplot(131), plt.imshow(img_1, cmap='gray')
plt.title("Threshold 200"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_box_1, cmap='gray')
plt.title("Threshold 200 (Box)"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_gaus_1, cmap='gray')
plt.title("Threshold 200 (Gauss)"), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(10,3))
plt.subplot(131), plt.imshow(img_2, cmap='gray')
plt.title("Threshold 800"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_box_2, cmap='gray')
plt.title("Threshold 800 (Box)"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_gaus_2, cmap='gray')
plt.title("Threshold 800 (Gauss)"), plt.xticks([]), plt.yticks([])

plt.show()