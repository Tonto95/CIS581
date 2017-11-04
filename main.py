import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from homography import homographize
from stitch import stitch


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
	# warped_pics = pics

	stitcher = cv2.createStitcher(False)


	for i in range(3):
		warped_pics[i,:,:,:] = cylindrical_warp(pics[i],f)

	for i in range(2):
		if i == 0:
			src_pts,dst_pts = find_correspondances(warped_pics[i], warped_pics[i+1])
			H, mask = homographize(src_pts, dst_pts)

			final_img = stitch(warped_pics[i], warped_pics[i+1], H)
		else:
			src_pts,dst_pts = find_correspondances(final_img, warped_pics[i+1])
			H, mask = homographize(src_pts, dst_pts)

			final_img = stitch(final_img, warped_pics[i+1], H)
			# result = stitcher.stitch((np.array(final_img, dtype = np.uint8), np.array(warped_pics[i+1], dtype = np.uint8)))


	# result = stitcher.stitch((np.array(warped_pics[0], dtype = np.uint8), np.array(warped_pics[1], dtype = np.uint8)))


	# matches = find_correspondances(warped_pics[0], warped_pics[1])

	imgplot = plt.imshow(final_img)
	plt.show()
	# cv2.imshow("correspondences", final_img)
	# cv2.waitKey()



