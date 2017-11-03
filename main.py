import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from homography import homographize


# Takes in a string path and the dimensions of the picture arrays
def master(path, size):
    pictures = image_loader.loader(path, size)
    return pictures


if __name__ == '__main__':
	path = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs"
	size = np.array([1632,1224,3])

	pics = master(path, size)

	f = (1224*4.15)/(4.8)

	warped_pics = np.zeros((pics.shape[0], int(f), int(f),3))

	stitcher = cv2.createStitcher(False)

	# src_pts,dst_pts = find_correspondances(pics[0], pics[1])

	# H, mask = homographize(src_pts, dst_pts)

	# print mask
	for i in range(2):
		warped_pics[i,:,:,:] = cylindrical_warp(pics[i],f)

	result = stitcher.stitch((np.array(warped_pics[0], d_type = np.uint8), np.array(warped_pics[1], d_type = np.uint8)))


	# matches = find_correspondances(warped_pics[0], warped_pics[1])






	cv2.imshow("cylindrical projection", result[1])
	cv2.waitKey()




