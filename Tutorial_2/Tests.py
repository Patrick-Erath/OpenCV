import numpy as np 
import matplotlib.pyplot as plt
import cv2

img_path = "./Pictures/RawPicture.jpg"
img = cv2.imread(img_path)
length = 9

res1 = cv2.medianBlur(img, length)

res2 = cv2.GaussianBlur(img, (length, length), 0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)

plt.subplot(131), plt.imshow(img)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res1)
plt.title("Median"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res2)
plt.title("Gaussian"), plt.xticks([]), plt.yticks([])
plt.show()