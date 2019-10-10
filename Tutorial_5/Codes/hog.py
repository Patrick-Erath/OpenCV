import cv2
import numpy as np
import matplotlib.pyplot as plt 

path = cv2.imread("bird.png")
img = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

cell_size = (8, 8)		# height x width in pixels
block_size = (2, 2)		# height x width in cells
nbins = 9				# number of orientation bins

# STEP 1 : Preprocessing
# winSize is the size of the image cropped to multiple of the cell size
# all arguments should be given in terms of number of pixels
winX = img.shape[1] // cell_size[1] * cell_size[1]
winY = img.shape[0] // cell_size[0] * cell_size[0]

blockX = block_size[1] * cell_size[1]
blockY = block_size[0] * cell_size[0]

print("Block_X:", blockX)
print("Block_Y:", blockY)
print("Image Shape:", img.shape)
print("Blocks:", img.shape[1]//8-1 + img.shape[0]//8-1)

# create HoG object
hog = cv2.HOGDescriptor(_winSize = (winX, winY),
						_blockSize = (blockX, blockY),
						_blockStride = (cell_size[1], cell_size[0]),
						_cellSize = (cell_size[1], cell_size[0]),
						_nbins = nbins
						)
n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])

# Computer HoG features
hog_feats = hog.compute(img) \
				.reshape(n_cells[1] - block_size[1] + 1,
						 n_cells[0] - block_size[0] + 1,
						 block_size[1], block_size[0], nbins) \
				.transpose((1, 0, 3, 2, 4))  # index blocks by row first

# hog_feats.shape = (38, 58, 2, 2, 9)

# Below, we take 38x58 (:, :), then change the block number

# Preview
plt.figure(figsize = (13,3))
plt.subplot(151), plt.imshow(img, cmap='gray')
plt.title("Original Image"), plt.xticks([]), plt.yticks([])

plt.subplot(152)
plt.pcolor(hog_feats[:, :,  0,0,0]) # hog_feats.shape = (38, 58)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("HOG bin = 0, block = 1"), plt.xticks([]), plt.yticks([])

plt.subplot(153)
plt.pcolor(hog_feats[:, :, 0,1,0])  # hog_feats.shape = (38, 58)	( : from beg->end)
plt.gca().invert_yaxis() # gca -> get current axis
plt.gca().set_aspect('equal', adjustable='box') # set box aspect 1:1
plt.title("HOG bin = 0, block = 2"), plt.xticks([]), plt.yticks([])

plt.subplot(154)
plt.pcolor(hog_feats[:, :, 1,0,0])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("HOG bin = 0, block = 3"), plt.xticks([]), plt.yticks([])

plt.subplot(155)
plt.pcolor(hog_feats[:, :, 1,1,0])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("HOG bin = 0, box = 4"), plt.xticks([]), plt.yticks([])
plt.colorbar(fraction=0.04)

plt.show()