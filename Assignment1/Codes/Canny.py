import cv2
import matplotlib.pyplot as plt 
import numpy as np

img_path = '../dolphin.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
height, width = img.shape

K = [5,9,13]
L = [10,30,50]
H = [100,150, 200]

Combinations = np.empty([27,3])
Titles = ["" for x in range(27)]

i = 0
for k in K:
    for l in L:
        for h in H:
            Combinations[i]=[k,l,h]
            Titles[i] = "K:"+str(k)+" L:"+str(l)+" H:"+str(h)
            i+=1

            
imgs = np.empty([27, height, width])
    
for i in range(0, 27):
    k = int(Combinations[i][0])
    l = int(Combinations[i][1])
    h = int(Combinations[i][2])
    img_gaussian = cv2.GaussianBlur(img, (k, k), 0)
    imgs[i] = cv2.Canny(img_gaussian, l, h)

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[0], cmap='gray')
plt.title(Titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[1], cmap='gray')
plt.title(Titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[2], cmap='gray')
plt.title(Titles[2]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[3], cmap='gray')
plt.title(Titles[3]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[4], cmap='gray')
plt.title(Titles[4]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[5], cmap='gray')
plt.title(Titles[5]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[6], cmap='gray')
plt.title(Titles[6]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[7], cmap='gray')
plt.title(Titles[7]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[8], cmap='gray')
plt.title(Titles[8]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[9], cmap='gray')
plt.title(Titles[9]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[10], cmap='gray')
plt.title(Titles[10]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[11], cmap='gray')
plt.title(Titles[11]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[12], cmap='gray')
plt.title(Titles[12]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[13], cmap='gray')
plt.title(Titles[13]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[14], cmap='gray')
plt.title(Titles[14]), plt.xticks([]), plt.yticks([])


plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[15], cmap='gray')
plt.title(Titles[15]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[16], cmap='gray')
plt.title(Titles[16]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[17], cmap='gray')
plt.title(Titles[17]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[18], cmap='gray')
plt.title(Titles[18]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[19], cmap='gray')
plt.title(Titles[19]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[20], cmap='gray')
plt.title(Titles[20]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[21], cmap='gray')
plt.title(Titles[21]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[22], cmap='gray')
plt.title(Titles[22]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[23], cmap='gray')
plt.title(Titles[23]), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,3))
plt.subplot(131), plt.imshow(imgs[24], cmap='gray')
plt.title(Titles[24]), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgs[25], cmap='gray')
plt.title(Titles[25]), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgs[26], cmap='gray')
plt.title(Titles[26]), plt.xticks([]), plt.yticks([])

plt.show()