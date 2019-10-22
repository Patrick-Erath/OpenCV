# Compute a homography to align the images using RANSAC method 
# Arrange matching keypoints in two seperate lists

# Arrange matching keypoints into two seperate lists
GoodMatches = []
for i, m in enumerate(matches):
    if m.distance < 300:
        GoodMatches.append((m.trainIdx, m.queryIdx))

# Get the keypoints that are good matches
m_crop = np.float32([ keypoints_crop[i].pt for(__, i) in GoodMatches])
m_occ = np.float32([ keypoints_occ[i].pt for (i, __) in GoodMatches])

# homography
H = []

for i in range(50):
    H_temp,_ = cv2.findHomography(m_crop, m_occ, cv2.RANSAC, i)
    H.append(H_temp)

height_c, width_c = book_crop.shape[:2]

# and apply the transformation on the reference image
for i in range(50):
    wimg = cv2.warpPerspective(book_crop, H[i], (width_c, height_c))
    plt.figure(figsize=(10,10))
    plt.imshow(wimg), plt.xticks([]), plt.yticks([])
    plt.show()
    print(i)