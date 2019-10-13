import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# Generate data
data = np.float32(
		np.vstack((		
			np.random.normal(loc=np.array([5,5]), scale=np.array([3,2]), size=(1000,2)),
			np.random.normal(loc=np.array([-5,-5]), scale=np.array([4,1]), size=(1000,2))
			))
		)
# np.float32 -> data type
# np.vstack -> stack arrays in sequence vertically
# np.random.normal -> draw random samples from a normal (Gaussian) distribution
	# @params: loc = mean (center) of distribution, scale  = std (width) of distibution, size = shape of distribution

# Before calling openCV's kmeans, we need to specify the stop criteria
# stop when either 100 iterations or an accuracy of 1 is reached
# define criteria = (type, max_iter=100, epsilon=1.0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

# set number of clusters
K = 2

# apply Kmeans
# cv2.kmeans(data, K, bestLabels, criteria, attempTs, flags[, centers]) -> retval,  bestLabels, centers
# compactness : the sum of squared distance from each point to their corresponding centers
# labels: : the label array (each element marked '0', '1', '3', etc)
# centers: array of centers of clusters

compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags=cv2.KMEANS_RANDOM_CENTERS)

# Can easily split the data into different clusters depending on their labels
A = data[labels.ravel()==0]
B = data[labels.ravel()==1]

plt.figure(figsize=(5,5))
plt.subplot(121)
plt.plot(data[:,0], data[:,1], "ko")
plt.title("Data points")
plt.subplot(122)
plt.scatter(A[:,0], A[:,1], color='b')
plt.scatter(B[:,0], B[:,1], color='r')
plt.scatter(centers[:,0], centers[:,1], s=80, color='y', marker='*')
plt.title("Data points - Labeled")
plt.show()