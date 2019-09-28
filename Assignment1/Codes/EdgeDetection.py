import cv2
import matplotlib.pyplot as plt 
import numpy as np
from Sharpening import *

kernel = np.ones((5, 5), np.float32)/(5*5)
# Applying intensity channels
I_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
I_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

I_x_sharp = cv2.Sobel(sharp_box, cv2.CV_64F, 1, 0, ksize=5)
I_y_sharp = cv2.Sobel(sharp_gaus, cv2.CV_64F, 0, 1, ksize=5)

plt.figure(figsize=(10, 3))
plt.subplot(121), plt.imshow(I_x, cmap='gray')
plt.title("X-axis Edge Detection"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I_x_sharp, cmap='gray')
plt.title('X-axis Edge detection (S)'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(10, 3))
plt.subplot(121), plt.imshow(I_y, cmap='gray')
plt.title("Y-axis Edge Detection"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I_y_sharp, cmap='gray')
plt.title('Y-axis Edge detection (S)'), plt.xticks([]), plt.yticks([])

plt.show()