import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img_path = "./biosphere.jpg"

# Load image as grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, (7, 7), 0)

# compute Canny edges
I_edge = cv2.Canny(img, 10, 50)
#I_edge = cv2.Canny(img, 50, 100)


# Canny edge : filters out noise then detects

# display images
plt.figure(figsize=(5, 5))
plt.subplot(211), plt.imshow(img, cmap="gray")
plt.title("Image"), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(I_edge, cmap="gray")
plt.title("I_edge"), plt.xticks([]), plt.yticks([])
plt.show()