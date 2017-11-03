'''
	Authors: Ahmed Zahra, Anthony Owusu
	CIS 581
'''

import cv2 
import numpy as np
import math

'''
	Projects x,y image coordinates onto a cylinder
		- Inputs:
			- x,y coordinates
			- f - focal lenght of camera
			- x_c,y_c - center of image
		- Outputs:
			- x_cyl,y_cyl - cylindrical coordinate projection
'''
def get_cylindrical_coordinates(x,y,f,x_c,y_c):
	# Calculate the parameters theta and h
	theta = np.arcsin(x/(np.sqrt(x*x + f*f)));
	h = y/(np.sqrt(x*x + f*f));

	# Calculate the x and y cylindrical prjections
	x_cyl = f*theta + x_c
	y_cyl = f*h + y_c

	return int(x_cyl), int(y_cyl)

'''
	Projects image onto cylinder
		- Inputs:
			- img: the img to be projected
			- f - focal lenght of camera

		- Outputs:
			- img_warped: the warped image
'''
def cylindrical_warp(img,f):
	h,w,p = img.shape

	# Get image center coordinates
	x_c = np.floor(w/2.0)
	y_c = np.floor(h/2.0)

	img_warped = np.zeros((int(f),int(f),3))

	for i in range(w):
		for j in range(h):
			curr_x = i - x_c
			curr_y = j - y_c 

			ii, jj = get_cylindrical_coordinates(curr_x,curr_y,f,x_c,y_c)

			if ii >= f-1:
				ii = int(f-1)

			if jj >= f-1:
				jj = int(f-1)	

			img_warped[jj,ii,:] = img[j,i,:]


	return np.array(img_warped, dtype = np.uint8)