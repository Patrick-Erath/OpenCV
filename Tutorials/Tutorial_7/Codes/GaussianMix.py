import cv2 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn import mixture

# Create sample train data
traind = np.float32(
			np.vstack((
				np.random.normal(loc=np.array([5,5]), scale=np.array([3,2]), size=(1000,2)),
				np.random.normal(loc=np.array([-5,-5]), scale=np.array([4,1]), size=(1000,2))
			))
		)

# Create sample test data
testd = np.float32(
			np.vstack((
					np.random.normal(loc=np.array([5,5]), scale=np.array([3,2]), size=(100,2)),
					np.random.normal(loc=np.array([-5,-5]), scale=np.array([4,1]), size=(100,2))
				))
		)

# define criteria, number of clusters (K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
compactness, labels, centers = cv2.kmeans(traind, 2, None, criteria, 10, flags=cv2.KMEANS_RANDOM_CENTERS)
 
A = traind[labels.ravel()==0]
B = traind[labels.ravel()==1]

plt.figure(figsize=(10,5))
plt.subplot(143)
plt.plot(testd[:,0], testd[:,1],"ko")
plt.title("Test data points")
plt.subplot(144)
plt.scatter(A[:,0], A[:,1], color='b')
plt.scatter(B[:,0], B[:,1], color='r')
plt.scatter(centers[:,0], centers[:,1], s=80, color='y', marker='*')
plt.title("Test data points - Labeled")
plt.suptitle("K-means")
plt.show()