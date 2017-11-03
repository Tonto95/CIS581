'''
	Authors: Ahmed Zahra, Anthony Owusu
	CIS 581
'''
import cv2
import numpy as np

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
	descriptors, keypoints = sift_obj.detectAndComput(img_gray, None)

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
	# Extract the features from both images
	des1, kp1 = extract_features(img1)
	des2, kp2 = extract_features(img2)

	# Initialize FLANN algorithm parameters to be used in the feature matching process 
	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# For testing: Display the images with correspondences 
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
	cv2.imshow("correspondences", img3)
	cv2.waitKey()

	return matches
