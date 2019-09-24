import cv2
import matplotlib.pyplot as plt
import numpy as np

img_clean_path = "./Circles.png"
img_gauss_path = "./Circles_gauss.png"

# Read image
img_noisy = cv2.imread(img_gauss_path)

# Filter image using 5x5, sigmaX = 0 
res1 = cv2.GaussianBlur(img_noisy, (5,5), 0) 

# Convert to RGB for matplotlib
res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32)/(5*5)

# Filter using Box
res2 = cv2.filter2D(img_noisy, -1, kernel)

# Convert from BGR to RGB
res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(res2)
plt.title("Circles w/ Box 5x5"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res1)
plt.title("Circles w/ Gaussian 5x5"), plt.xticks([]), plt.yticks([])
plt.show()
