import cv2
import numpy as np
import matplotlib.pyplot as plt 

####################
# FEATURE MATCHING #
####################

# load image
img1 = cv2.imread("../bird.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

height, width = img1.shape[0:2] # image size

# Create a rotatedm scaled duplicte
# Rotation matrix around the center pixel, 30 degrees, scale of 1.2
M = cv2.getRotationMatrix2D((width/2,height/2), 30, 1.2)
# Apply Transformation Matrix
img2 = cv2.warpAffine(img1, M, (width, height))

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Compute the keypoints / descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create a BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort descriptors
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches
img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=2)

plt.figure(figsize=(6,3))
plt.imshow(img3)
plt.title("Matched Keypoints"), plt.xticks([]), plt.yticks([])
plt.show()