import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.future import graph
from skimage.segmentation import quickshift

# Read image
img = cv2.cvtColor(cv2.imread('../pyramids.jpg'), cv2.COLOR_BGR2RGB)

# Mean shift segmentation
label = quickshift(img, max_dist=20)
out = color.label2rgb(label, img, kind='avg')

# display
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(out)
plt.title("Segmented Image"), plt.xticks([]), plt.yticks([])
plt.show()

# Graph Cut Segmentation

# apply k-means, this will generate super pixels
labels1 = segmentation.slic(img, compactness=5, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')
plt.imshow(out1); plt.show()

g = graph.rag_mean_color(img, labels1, mode='similarity')

labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg')

# display
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(out2)
plt.title("Segmented Image"), plt.xticks([]), plt.yticks([])
plt.show()