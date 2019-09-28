import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "./numbers.jpg"
img = cv2.imread(image_path)
#height, width, thick = img.shape

# conver to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape

THRESHOLDS = [55, 90, 150]

# Create placeholder matrix for the outputs
imgs_thres = [0, 0, 0]
for i in range(3):
    imgs_thres[i] = np.zeros((height, width), dtype=np.uint8)
    
# same operation w/o loops
ret1, img_thres1 = cv2.threshold(img_gray, THRESHOLDS[0], 255, cv2.THRESH_BINARY)
ret2, img_thres2 = cv2.threshold(img_gray, THRESHOLDS[1], 255, cv2.THRESH_BINARY)
ret3, img_thres3 = cv2.threshold(img_gray, THRESHOLDS[2], 255, cv2.THRESH_BINARY)

# display
plt.subplot(141), plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale Image"), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(img_thres1, cmap='gray')
plt.title("Threshold = 55"), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(img_thres2, cmap='gray')
plt.title("Threshold = 90"), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(img_thres3, cmap='gray')
plt.title("Threshold = 150"), plt.xticks([]), plt.yticks([])
plt.show()

