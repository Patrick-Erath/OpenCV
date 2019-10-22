import numpy as np 
import cv2
import matplotlib.pyplot as plt

# Read in testing images & resize
def readImages(img_len, img_name):
	testing_imgs = []
	try:
		for i in range(0,img_len):
			img_temp = cv2.imread(img_name+str(i)+".jpeg")
			img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
			img_temp = cv2.resize(img_temp, (128,128))
			testing_imgs.append(img_temp)
	except():
		print('error reading')

	return testing_imgs

# Function for reading 
def hog_features(imgs_arr, cell_size=(4,4), block_size=(2,2), nbins=9, train=True):
	hog_feats_arr = []
	count = 0
	feats_sum = 0
	for img in imgs_arr:
		# Check window size
		if(img.shape[0]%16!=0 or img.shape[1]%16!=0):
			raise Exception('Invalid Image Size')
		else:
			# Compute window size
			win_XY = img.shape[0] // cell_size[0] * cell_size[1]

			# Compute blocks
			block_XY = block_size[0] * cell_size[0]

			# Create HoG object
			hog = cv2.HOGDescriptor(_winSize = (win_XY, win_XY),
									_blockSize = (block_XY, block_XY),
									_blockStride = (cell_size[1], cell_size[0]),
									_cellSize = (cell_size[1], cell_size[0]),
									_nbins = nbins
									)

			# Compute number of cells 
			n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])

			# Compute HoG features
			hog_feats = hog.compute(img) \
									.reshape(n_cells[1] - block_size[1] + 1,
										n_cells[0] - block_size[0] +1, 
										block_size[1], block_size[0], nbins) \
									.transpose((1, 0, 3, 2, 4))

			hog_feats_arr.append(hog_feats)

	if train:
		for i in range(0, len(hog_feats_arr)):
			feats_sum += hog_feats_arr[i]
		return feats_sum/14
	else:
		return hog_feats_arr


def calculateDistance(mean_hog, testing_imgs):
	distances = []
	for i in range(0, len(testing_imgs)):
		dist_temp = np.linalg.norm(testing_imgs[i]-mean_hog)
		distances.append(dist_temp)

	return distances


if __name__ == '__main__':
	train_imgs = readImages(11, "./training_imgs/")
	test_imgs = readImages(2, "./testing_imgs/")
	false_imgs = readImages(3, "./false_testing/")

	hog_feats_mean = hog_features(train_imgs)
	hog_feats_test = hog_features(test_imgs, train=False)
	hog_feats_false = hog_features(false_imgs, train=False)

	distance_tests = calculateDistance(hog_feats_mean, hog_feats_test)
	distance_false = calculateDistance(hog_feats_mean, hog_feats_false)

	# TODO : threshold

	for dis in distance_tests:
		print("distance test : ",dis)

	print('-'*30)
	

	for dis in distance_false:
		print("distance FALSE test : ", dis)














