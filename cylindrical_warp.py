'''
	Authors: Ahmed Zahra, Anthony Owusu
	CIS 581
'''

import cv2 
import numpy as np
import math


def get_cylindrical_coordinates(x,y,f,x_c,y_c):
	theta = np.arcsin(curr_x/(np.sqrt(curr_x*curr_x + f*f)));
	h = curr_y/(np.sqrt(curr_x*curr_x + f*f));

	x_cyl = f*theta + x_c
	y_cyl = f*h + y_c

	return int(x_cyl), int(y_cyl)

def cylindrical_warp(img,f):
	h,w,p = img.shape

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