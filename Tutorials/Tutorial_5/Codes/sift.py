import numpy as np 
import cv2
import matplotlib.pyplot as plt 

###########################
# SIFT Feature Detector #
#########################

# Load image 
img = cv2.imread("../bird.png")
# Make a copy
img_disp = img.copy()

# Convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_disp = cv2.cvtColor(img_disp, cv2.cv2.COLOR_BGR2RGB)

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Detect SIFT features with no masks
keypoints = sift.detect(img, None)

# Draw the keypoints
cv2.drawKeypoints(img, keypoints, img_disp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display
plt.figure(figsize=(10,3))
plt.subplot(121), plt.imshow(img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_disp)
plt.title("SIFT features"), plt.xticks([]), plt.yticks([])
plt.show()

# number of SIFT keypoints
print("Number of keypoints: \t" + str(len(keypoints)))