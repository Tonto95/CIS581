import os
import sys
import cv2
import math
import numpy as np

from numpy import linalg

output_dir = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs"


def findDimensions(image, homography):
	base_p1 = np.ones(3, np.float32)
	base_p2 = np.ones(3, np.float32)
	base_p3 = np.ones(3, np.float32)
	base_p4 = np.ones(3, np.float32)

	(y, x) = image.shape[:2]

	base_p1[:2] = [0,0]
	base_p2[:2] = [x,0]
	base_p3[:2] = [0,y]
	base_p4[:2] = [x,y]

	max_x = None
	max_y = None
	min_x = None
	min_y = None

	for pt in [base_p1, base_p2, base_p3, base_p4]:

		hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
		
		hp_arr = np.array(hp, np.float32)

		normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

	if ( max_x == None or normal_pt[0,0] > max_x ):
		max_x = normal_pt[0,0]

		if ( max_y == None or normal_pt[1,0] > max_y ):
			max_y = normal_pt[1,0]

		if ( min_x == None or normal_pt[0,0] < min_x ):
			min_x = normal_pt[0,0]

		if ( min_y == None or normal_pt[1,0] < min_y ):
			min_y = normal_pt[1,0]

	min_x = min(0, min_x)
	min_y = min(0, min_y)
	
	return (min_x, min_y, max_x, max_y)

def stitch(base_img, next_img, H): 

	H_inv = linalg.inv(H)
	(min_x, min_y, max_x, max_y) = findDimensions(next_img, H_inv)
    
	# Adjust max_x and max_y by base img size
	max_x = max(max_x, base_img.shape[1])
	max_y = max(max_y, base_img.shape[0])
    
	move_h = np.matrix(np.identity(3), np.float32)
    
	if ( min_x < 0 ):
		move_h[0,2] += -min_x
		max_x += -min_x
    
	if ( min_y < 0 ):
		move_h[1,2] += -min_y
		max_y += -min_y
    
	print "Homography: \n", H
	print "Inverse Homography: \n", H_inv
	print "Min Points: ", (min_x, min_y)
    
	mod_inv_h = move_h * H_inv
    
	img_w = int(math.ceil(max_x))
	img_h = int(math.ceil(max_y))
    
	print "New Dimensions: ", (img_w, img_h)
    
	# Warp the new image given the homography from the old image
	base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h))
	print "Warped base image"
    
	# utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
	# cv2.destroyAllWindows()
    
	next_img_warp = cv2.warpPerspective(next_img, mod_inv_h, (img_w, img_h))
	print "Warped next image"

    
    # Put the base image on an enlarged palette
	enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    
	print "Enlarged Image Shape: ", enlarged_base_img.shape
	print "Base Image Shape: ", base_img.shape
	print "Base Image Warp Shape: ", base_img_warp.shape
    
	# enlarged_base_img[y:y+base_img.shape[0],x:x+base_img.shape[1]] = base_img
    # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp
    
    # Create a mask from the warped image for constructing masked composite
	(ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp.astype(np.uint8), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
   
	enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
    
	# Now add the warped image
	final_img = cv2.add(enlarged_base_img, next_img_warp, 
		dtype=cv2.CV_8U)
    
 #    # Crop off the black edges
	# final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
	# _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
	# contours,_,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# print "Found %d contours..." % (len(contours))
    
	# max_area = 0
	# best_rect = (0,0,0,0)

	# print contours
    
	# for cnt in contours:
	# 	x,y,w,h = cv2.boundingRect(cnt)
    
	# 	deltaHeight = h-y
	# 	deltaWidth = w-x
    
	# 	area = deltaHeight * deltaWidth
    
	# 	if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
	# 		max_area = area
	# 		best_rect = (x,y,w,h)
    
	# if ( max_area > 0 ):
	# 	print "Maximum Contour: ", max_area
	# 	print "Best Rectangle: ", best_rect
    
	# final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
	# 	best_rect[0]:best_rect[0]+best_rect[2]]
	# final_img = final_img_crop

	return final_img