import cv2
import matplotlib.pyplot as plt
import numpy as np 

image_path = "./Pictures/coloredChips.png"
img = cv2.imread(image_path)

# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape

# set the threshold
THRESH = 130

# create placeholder matrix for the output
img_thres = np.zeros((height, width), dtype=np.uint8)

for row in range(0, height):
	for col in range(0, width):
		if img_gray[row, col] > THRESH:
			img_thres[row, col] = 255
		# else already = 0 from initializing the array

# same operation without loops
ret, img_thres2 = cv2.threshold(img_gray, THRESH, 255, cv2.THRESH_BINARY)

# display
plt.subplot(131), plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale Image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_thres, cmap="gray")
plt.title("Thresholded Image"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_thres2, cmap="gray")
plt.title("Thresholded Image2"), plt.xticks([]), plt.yticks([])
plt.show()