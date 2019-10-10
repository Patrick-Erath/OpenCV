import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

####################
# SIFT DESCRIPTORS #
####################

# Load image
img = cv2.imread("../bird.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a sift object
sift = cv2.xfeatures2d.SIFT_create()

# Detect SIFT featues with no masks
keypoints = sift.detect(img, None)
print("Number of keypoints: \t", len(keypoints))

# Computer SIFT descriptors
keypoints, descriptors = sift.compute(img, keypoints)
print("Descriptor size: \t",  descriptors.shape)
# Can use keypoints, descriptors = sift.detectAndComputer(img)

# Plot some of the 1x128 SIFT descriptors
for i in range(4):
	plt.plot(descriptors[i])
	plt.ylim(-5, 150)
	plt.xlim([0,128])
plt.title("Magnitude of each descriptor")
plt.xlabel('pixel number')
plt.ylabel('magnitude')
plt.show()