import cv2
import matplotlib.pyplot as plt 
import numpy as np

# image filtering
image_path = "./Pictures/board.tif"
img = cv2.imread(image_path)

# create a box filter
kernel = np.ones((5,5), np.float32)/(5*5)
print("kernel:\t", kernel)
# Kernel sets equal weight to all the pixels in the kernel, thus 
# convlolution will take the average of the pixelx

if img.size != None:
	print('image loaded correctly')

# filter the image
res1 = cv2.filter2D(img, -1, kernel)  # depth = -1 ie same

# similar results using cv2.blur()
res2 = cv2.blur(img, (5,5))

# convert from BGR to RGB for dfisplaying
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

# plot
plt.subplot(131), plt.imshow(img)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res1)
plt.title('Box filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res2)
plt.title('Blur filter'), plt.xticks([]), plt.yticks([])
plt.show()

# https://edoras.sdsu.edu/doc/matlab/toolbox/images/linfilt4.html