import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import segmentation, color


img = cv2.cvtColor(cv2.imread('../home.jpg'), cv2.COLOR_BGR2RGB)

# slic(image, n_segments=100, compactness=10.0, max_iter=10,
	#  sigma=0, spacing=None, multichannel=True, convert2lab=None
	#  enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False
	#)
# n_segments: the (approx) number of labels in the segmented output image
# compactness: balaces color & space proximity. Higher compactness -> more weight to space proximity
# max_iter: maximium number of iterations of k-means
# sigma: width of gaussian smoothing kernel
# spacing: voxel spacing along each image dimension
# multichannel: Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension.
# enforce_connectivity: Whether the generated segments are connected or not
# min_size_factor: Proportion of the minimum segment size to be removed
# max_size_factor: Proportion of the maximum connected segment size.
   
labels = segmentation.slic(img, compactness=10, n_segments=3)
out = color.label2rgb(labels, img, kind='avg')

plt.figure(figsize=(5,3))
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(out)
plt.title("Segmented Image"), plt.xticks([]), plt.yticks([])
plt.show()