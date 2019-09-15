import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# load image
image_path = "./Pictures/office_2.jpg"
img = cv2.imread(image_path)
img2 = img.copy() 		# deep copy of image (ie copy original before modifying)
height, width, depth = img.shape	# reading image size

t0 = time.time()
for row in range(0, height):
	for col in range(0, width):
		for ch in range(0, depth):
			img2[row, col, ch] += 50  	# bridghten image
print("Elapsed time: \t", time.time() - t0)		# this way is not the fastest

# convert from BGR to RGB for displaying
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img) 	# 1x2 subplot, 1 selected
plt.title("Image1"), plt.xticks([]), plt.yticks([])  # remove axes ticks (numbering)
plt.subplot(122), plt.imshow(img2)
plt.title("Image2"), plt.xticks([]), plt.yticks([])

plt.show()  # show both images