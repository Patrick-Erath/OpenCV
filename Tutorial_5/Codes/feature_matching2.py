import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img_left = cv2.imread("../S1.jpg")
img_right = cv2.imread("../S2.jpg")

img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Create SIFT object and find descriptors
sift = cv2.xfeatures2d.SIFT_create()
keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

# Create a BF Matcher object to find matches
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(descriptors_right, descriptors_left)

# Sort mathces
matches = sorted(matches, key = lambda x:x.distance)

# Draw keypoints
imgmatch = cv2.drawMatches(img_right, keypoints_right, img_left, keypoints_left, matches[:10], None, flags=2)

# Arrange matching keypoints in two seperate lists
GoodMatches = []
for i, m in enumerate(matches):
	if m.distance < 1000:
		GoodMatches.append((m.trainIdx, m.queryIdx))

# Get the  keypoints that are good matches
mpr = np.float32([ keypoints_right[i].pt for (__, i) in GoodMatches])
mpl = np.float32([ keypoints_left[i].pt for (i, __) in GoodMatches])

# Find homography and wrap image accordingly
H, __ = cv2.findHomography(mpr, mpl, cv2.RANSAC, 4)
wimg = cv2.warpPerspective(img_right, H, (img_right.shape[1]+img_left.shape[1], img_right.shape[0]))
wimg[:,:img_left.shape[1],:] = img_left

plt.figure(figsize=(8,3))
plt.subplot(121)
plt.imshow(imgmatch)
plt.title("Matches keypoints"), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(wimg)
plt.title("Panoramic image"), plt.xticks([]), plt.yticks([])
plt.show()