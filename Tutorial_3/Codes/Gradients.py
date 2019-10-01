import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img_path = "../biosphere.jpg"

# Load image as grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply 15x15 Sobel filtrers to the intensity channel
I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15) #src, float, xorder=1, yorder=0, kernel size
I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)
# Computer sobel Magnitude / Phase
I_m = cv2.magnitude(I_x, I_y)
I_p = cv2.phase(I_x, I_y)

# display images
plt.figure(figsize=(10, 3))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(I_x, cmap='gray')
plt.title("I_x"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(I_y, cmap='gray')
plt.title("I_y"), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(10, 3))
plt.subplot(121), plt.imshow(I_m, cmap='gray')
plt.title("I_m"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I_p, cmap='gray')
plt.title("I_p"), plt.xticks([]), plt.yticks([])
plt.show()
#plt.show()