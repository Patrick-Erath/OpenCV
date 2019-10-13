import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.cvtColor(cv2.imread('../home.jpg'), cv2.COLOR_BGR2RGB)

# Make height & width pixel --> 1  array
Z = img.reshape((-1,3))

# Convert to np.float32
Z = np.float32(Z)

# Define criteria, number of clusters (K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8 and make original image
center = np.uint8(center)
out = center[label.flatten()]
out = out.reshape((img.shape))

plt.figure(figsize=(7,3))
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(out)
plt.title("Segmentated Image"), plt.xticks([]), plt.yticks([])
plt.show()