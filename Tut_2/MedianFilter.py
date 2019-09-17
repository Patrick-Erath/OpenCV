import numpy as np
import matplotlib.pyplot as plt
import cv2

# image filtering
image_path = "./Pictures/kids.tif"
img = cv2.imread(image_path)
filt = 3

# filter image using median filter (Get rid of salt/pepper noise)
res = cv2.medianBlur(img, filt)
res1 = cv2.GaussianBlur(img, (filt,filt), 0)

# convert from BGRG to RGB for displaying
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)

# plot
plt.subplot(131), plt.imshow(img)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res)
plt.title("Median Filter"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res)
plt.title("Gaussian Filter"), plt.xticks([]), plt.yticks([])
plt.show()