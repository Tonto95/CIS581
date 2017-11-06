import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from homography import homographize
from stitch import calculate_size, merge_images, crop

# Takes in a string path and the dimensions of the picture arrays
def main():
	path = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs_ordered/"
	f = 1012*0.25
	size = np.array([408, 306, 3])
	pictures = image_loader.loader(path, size)

	warped_pics = np.zeros((pictures.shape[0], int(f), int(f), 3))

	# Set this value to be >= 1 to change the starting image in the panorama
	start = 3

	# NUMBER TO CHANGE
	# set to >= 3 to stitch 2 or more images
	n = 4

	if start + n >= 15:
		print "This wont work"
		return

	for i in range(start, start+n):
		warped_pics[i,:,:,:] = cylindrical_warp(pictures[i], f)

	pictures = warped_pics

	main_image = pictures[start]


	for i in range(start+1, start+n):
		src_pts, dst_pts = find_correspondances(main_image, pictures[i])

		H, mask = homographize(src_pts, dst_pts)

		# H = H * last_H
		## Calculate size and offset of merged panorama.
		(size, offset) = calculate_size(main_image.shape, pictures[i].shape, np.linalg.inv(H))
		print "output size: %ix%i" % size

		## Finally combine images into a panorama.
		main_image = merge_images(main_image, pictures[i], H, size, offset, (src_pts, dst_pts))

	cv2.imwrite("/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/panorama.jpg", main_image)
	# plt.imshow(pictures[2], cmap='spring')
	# plt.show()

if __name__ == '__main__':
	main()
