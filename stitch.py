import numpy as np
import cv2
from numpy import linalg
import math

output_dir = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs"

# def stitch(base_img, next_img, H, xshift, yshift, pad):

# 	newImg = np.hstack((base_img,next_img))

# 	print newImg.shape

# 	count = 0

# 	for col in range(base_img.shape[1]+1,newImg.shape[1]):
# 		for row in range(pad,newImg.shape[0]-pad):
# 			if yshift >= 0:
# 				if col-xshift <= base_img.shape[1]:
# 	               # Feathering
# 					w = count / float(xshift)
# 					newImg[row - yshift, col - xshift, :] = np.floor(w*newImg[row, col,:] + (1-w)*base_img[row-yshift,col-xshift,:])
	                
# 				else:
# 					newImg[row - yshift, col - xshift, :] = newImg[row, col,:]
# 			else:
# 				if col-xshift <= base_img.shape[1]:
# 	               # Feathering
# 					w = count / float(xshift)
# 					newImg[row, col - xshift, :] = np.floor(w*newImg[row + yshift, col,:] + (1-w)*base_img[row,col-xshift,:])
	               
# 				else:
# 					newImg[row, col - xshift, :] = newImg[row + yshift, col,:]
	    
# 		count = count + 1;

# 	col = base_img.shape[1] - xshift + next_img.shape[1];
# 	newImg = newImg[:, 1:col,:];

def calculate_size(image1, image2, H):
  
	(h1, w1,_) = image1.shape
	(h2, w2,_) = image2.shape
  
	#remap the coordinates of the projected image onto the panorama image space
	top_left = np.dot(H,np.asarray([0,0,1]))
	top_right = np.dot(H,np.asarray([w2,0,1]))
	bottom_left = np.dot(H,np.asarray([0,h2,1]))
	bottom_right = np.dot(H,np.asarray([w2,h2,1]))

  
	#normalize
	top_left = top_left/top_left[2]
	top_right = top_right/top_right[2]
	bottom_left = bottom_left/bottom_left[2]
	bottom_right = bottom_right/bottom_right[2]

  
	pano_left = int(min(top_left[0], bottom_left[0], 0))
	pano_right = int(max(top_right[0], bottom_right[0], w1))
	W = pano_right - pano_left
  
	pano_top = int(min(top_left[1], top_right[1], 0))
	pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
	H = pano_bottom - pano_top
  
	size = (W, H)
  
  
	# offset of first image relative to panorama
	X = int(min(top_left[0], bottom_left[0], 0))
	Y = int(min(top_left[1], top_right[1], 0))
	offset = (-X, -Y)
      
	## Update the H to shift by the offset
	# does offset need to be remapped to old coord space?
	# print H
	# H[0:2,2] += offset

	return (size, offset)


def merge_images(image1, image2, H, size, offset):

	## TODO: Combine the two images into one.
	## TODO: (Overwrite the following 5 lines with your answer.)
	(h1, w1) = image1.shape[:2]
	(h2, w2) = image2.shape[:2]
  
	panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
	(ox, oy) = offset
  
	translation = np.matrix([
		[1.0, 0.0, ox],
		[0, 1.0, oy],
		[0.0, 0.0, 1.0]
	])
  
	H = translation * H
	# print H
  
	# draw the transformed image2
	cv2.warpPerspective(image2, H, size, panorama)
  
	panorama[oy:h1+oy, ox:ox+w1] = image1  
	# panorama[:h1, :w1] = image1  

	## TODO: Draw the common feature keypoints.

	return panorama