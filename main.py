import image_loader
import cv2
import numpy as np
from find_correspondances import find_correspondances
from cylindrical_warp import cylindrical_warp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from homography import homographize
from stitch import calculate_size, merge_images, merge_images_translation


# Takes in a string path and the dimensions of the picture arrays
def main():
	path = "/Users/Tonto/Documents/School/CIS/CIS519/CIS581/test_imgs"
	size = np.array([1632, 1224, 3])
	pictures = image_loader.loader(path, size)

	#warped_pics = np.zeros((pictures.shape[0], int(f), int(f), 3))

	src_pts,dst_pts = find_correspondances(pictures[1], pictures[0])

	H, mask = homographize(dst_pts, src_pts)

	## Calculate size and offset of merged panorama.
	(size, offset) = calculate_size(pictures[1].shape, pictures[0].shape, H)
	print "output size: %ix%i" % size

	## Finally combine images into a panorama.
	panorama = merge_images(pictures[1], pictures[0], H, size, offset, (src_pts, dst_pts))
	#panorama = merge_images_translation(pictures[1], pictures[0], offset)
	#cv2.imwrite("/Users/Tonto/Documents/School/CIS/CIS519/CIS581/panorama.jpg", panorama)
	#plt.imshow(panorama, cmap='spring')
	#plt.show()

	main_image = pictures[15]

	for i in reversed(range(0, 15)):
		src_pts, dst_pts = find_correspondances(main_image, pictures[i])

		H, mask = homographize(dst_pts, src_pts)

		## Calculate size and offset of merged panorama.
		(size, offset) = calculate_size(main_image.shape, pictures[i].shape, H)
		print "output size: %ix%i" % size

		## Finally combine images into a panorama.
		main_image = merge_images(main_image, pictures[i], H, size, offset, (src_pts, dst_pts))
		#panorama = merge_images_translation(pictures[1], pictures[0], offset)

	cv2.imwrite("/Users/Tonto/Documents/School/CIS/CIS519/CIS581/panorama.jpg", panorama)
	plt.imshow(main_image, cmap='spring')
	plt.show()

if __name__ == '__main__':
	main()