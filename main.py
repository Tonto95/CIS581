import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from homography import homographize
import stitch


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
	warped_pics = pics


	# for i in range(3):
	# 	warped_pics[i,:,:,:] = cylindrical_warp(pics[i],f)

	n = warped_pics.shape[0]
	n = n - 1
	im1 = warped_pics[0]

	for i in range(1):
		if i == 0:
			src_pts,dst_pts = find_correspondances(im1, warped_pics[n-i])
			H, _ = homographize(src_pts, dst_pts)

			size, offset = stitch.calculate_size(im1, warped_pics[n-i], H)
			# print size, offset
			final_img = stitch.merge_images(im1, warped_pics[n-i], H, size, offset)
			# final_img = stitch.stitch(im1, warped_pics[n-i], H, int(offset[0]), int(offset[1]), 0)

		else:
			src_pts,dst_pts = find_correspondances(final_img, warped_pics[n-i])
			H, _ = homographize(src_pts, dst_pts)

			size, offset = stitch.calculate_size(final_img, warped_pics[n-i], H)
			final_img = stitch.merge_images(final_img, warped_pics[n-i], H, size, offset)
			# final_img = stitch(final_img, warped_pics[i+1], H)


	imgplot = plt.imshow(np.array(final_img,dtype = np.uint8))
	plt.show()
	# cv2.imshow("correspondences", final_img)
	# cv2.waitKey()



