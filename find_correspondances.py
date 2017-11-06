'''
	Authors: Ahmed Zahra, Anthony Owusu
	CIS 581
'''
import cv2
import numpy as np
from stitch import calculate_size, merge_images, crop

MIN_MATCHES_COUNT = 4

'''
	This function takes in an image img, and returns a feature vector
		- Input:
				- img: N-by-M image
		- Output:
				- features: n-by-2 feature vector

'''
def extract_features(img):
	# Convert rgb image to gray
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


	sift_obj = cv2.xfeatures2d.SIFT_create()
	keypoints, descriptors = sift_obj.detectAndCompute(img_gray, None)

	return descriptors, keypoints


'''
	This function takes in two images img1, and img2, and returns an array of matches
		- Input:
				- img1: N-by-M image
				- img2: N-by-M image
		- Output:
				- matches: 

'''
def find_correspondances(img1, img2):
	# Convert images to cv images
	img1 = np.array(img1, dtype = np.uint8)
	img2 = np.array(img2, dtype = np.uint8)

	# Extract the features from both images
	des1, kp1 = extract_features(img1)
	des2, kp2 = extract_features(img2)

	# Initialize FLANN algorithm parameters to be used in the feature matching process 
	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.6*n.distance:
			good.append(m)
	# draw_params = dict(matchColor = (0,255,0),
 #                   singlePointColor = (255,0,0),
 #                   matchesMask = good,
 #                   flags = 0)

	# # For testing: Display the images with correspondences 
	# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
	# cv2.imwrite("/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/feature_matching.jpg", crop(np.array(img3,dtype = np.uint8)))

	if len(good) > MIN_MATCHES_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	return src_pts, dst_pts
