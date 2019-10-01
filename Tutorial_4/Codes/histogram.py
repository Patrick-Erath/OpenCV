import cv2
import numpy as numpy
import matplotlib.pyplot as plt 

# load the image
img = cv2.imread('../bird.png')
img_gray = cv2.imread('../bird.png', cv2.IMREAD_GRAYSCALE)
# 475 x 316

# computer histogram
# cv2.calcHist(image, channels, mask, histSize, ranges)
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# Compute histogram for each RGB channel
hist_color = []
for i in range(0, img.shape[2]):
	hist_color.append(cv2.calcHist([img], [i], None, [256], [0, 256]))

print("Image pixel count = " + str(img.shape[0]*img.shape[1]))
print("hist_gray pixel count = " + str(sum(hist_gray)))
print("hist_color[R] pixel count = " + str(sum(hist_color[0])))
print("hist_color[G] pixel count = " + str(sum(hist_color[1])))
print("hist_color[G] pixel count = " + str(sum(hist_color[2])))

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10,3))
plt.subplot(121), plt.imshow(img)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_gray, cmap='gray')
plt.title("GrayScale"), plt.xticks([]), plt.yticks([])
#plt.show()

# Display histograms
plt.figure(figsize=(10,3))
plt.subplot(121), plt.plot(hist_gray)
plt.title("Grayscale Histogram"), plt.xlim([0, 256]), plt.ylim(0)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.5)
plt.grid(which='minor', linestyle=':', linewidth=0.5)

plt.subplot(122)
colors = ['r', 'g', 'b']
for i in range(0, img.shape[2]):
	plt.plot(hist_color[i], color=colors[i])
plt.title("RGB Histogram"), plt.xlim([0,256]), plt.ylim(0)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.show()