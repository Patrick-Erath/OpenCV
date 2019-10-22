import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img_path = "./biosphere.jpg"

# Load image as grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply a 15x15 Laplacian filter to the intensity channel
I_lap = cv2.Laplacian(img, cv2.CV_32F, ksize=15)

# Laplacian filter is a derivative filter which detects edges by finding rapid changes

plt.figure(figsize=(5, 5))
plt.subplot(211), plt.imshow(img, cmap="gray")
plt.title("Image"), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(I_lap, cmap="gray")
plt.title("I_lap"), plt.xticks([]), plt.yticks([])
plt.show()