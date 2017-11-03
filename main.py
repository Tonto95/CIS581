import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp

# Takes in a string path and the dimensions of the picture arrays
def master(path, size):
    pictures = image_loader.loader(path, size)
    return pictures


if __name__ == '__main__':
	path = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs"
	size = np.array([1632,1224,3])

	pics = master(path, size)

	f = (1224*4.15)/(4.8)
	img_warped = cylindrical_warp(pics[0],f)

	cv2.imshow("cylindrical projection", img_warped)
	cv2.waitKey()



	# matches = find_correspondances(pics[0], pics[1])

