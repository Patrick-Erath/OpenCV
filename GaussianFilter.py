import numpy as np 
import matplotlib.pyplot as plt
import cv2

# read in the image
image_path = "./Pictures/board.tif"
img = cv2.imread(image_path)

filt1 = 5
filt2 = 15

# filter image using 5x5 Gaussian, the std is calculated automatically
res1 = cv2.GaussianBlur(img, (filt1, filt1), 0)  # img, kernel size, sigmaX
res2 = cv2.GaussianBlur(img, (filt2, filt2), 0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

plt.subplot(131), plt.imshow(img)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res1)
plt.title("3x3 Filter"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res2)
plt.title("9x9 Filter"), plt.xticks([]), plt.yticks([])

plt.show()