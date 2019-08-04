import numpy as np
import cv2
import os


watch_src = './Users/kingpatrick/Documents/git/OpenCV/files/watch.jpg'
img = cv2.imread(watch_src, cv2.IMREAD_COLOR)

cwd = os.getcwd()
print(cwd)
print(img)

#cv2.imshow('test', img)
#cv2.waitKey(0)
#cv2.destoyAllWindows()
