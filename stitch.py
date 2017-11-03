import cv2
import numpy as np

def stitch(img_list, H_list):
	final_img = img_list[0]

	for i,img_i in enumerate(img_list[1:]):
		H_i = H_list[i]

		# calculate startp 
		start_p = np.linalg.inv(H)*np.array([0,0,1])
		start_p = start_p/start_p[-1]

		# Transform the matrix
