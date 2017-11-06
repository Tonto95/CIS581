import cv2, numpy as np
import math
import image_loader

DEBUG = False

def findDimensions(image, homography):
	base_p1 = np.ones(3, np.float32)
	base_p2 = np.ones(3, np.float32)
	base_p3 = np.ones(3, np.float32)
	base_p4 = np.ones(3, np.float32)
    
	(y, x) = image.shape[:2]
    
	base_p1[:2] = [0,0]
	base_p2[:2] = [x,0]
	base_p3[:2] = [0,y]
	base_p4[:2] = [x,y]
    
	max_x = None
	max_y = None
	min_x = None
	min_y = None
    
	for pt in [base_p1, base_p2, base_p3, base_p4]:
    
		hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
    
		hp_arr = np.array(hp, np.float32)
    
		normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
    
		if ( max_x == None or normal_pt[0,0] > max_x ):
			max_x = normal_pt[0,0]
    
		if ( max_y == None or normal_pt[1,0] > max_y ):
			max_y = normal_pt[1,0]
    
		if ( min_x == None or normal_pt[0,0] < min_x ):
			min_x = normal_pt[0,0]
    
		if ( min_y == None or normal_pt[1,0] < min_y ):
			min_y = normal_pt[1,0]
    
	min_x = min(0, min_x)
	min_y = min(0, min_y)
    
	return (min_x, min_y, max_x, max_y)


def calculate_size(size_image1, size_image2, homography):
    (h1, w1) = size_image1[:2]
    (h2, w2) = size_image2[:2]

    # remap the coordinates of the projected image onto the panorama image space
    top_left = np.dot(homography, np.asarray([0, 0, 1]))
    top_right = np.dot(homography, np.asarray([w2, 0, 1]))
    bottom_left = np.dot(homography, np.asarray([0, h2, 1]))
    bottom_right = np.dot(homography, np.asarray([w2, h2, 1]))

    if DEBUG:
        print top_left
        print top_right
        print bottom_left
        print bottom_right

    # normalize
    top_left = top_left / top_left[2]
    top_right = top_right / top_right[2]
    bottom_left = bottom_left / bottom_left[2]
    bottom_right = bottom_right / bottom_right[2]

    if DEBUG:
        print np.int32(top_left)
        print np.int32(top_right)
        print np.int32(bottom_left)
        print np.int32(bottom_right)

    pano_left = int(min(top_left[0], bottom_left[0], 0))
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    W = pano_right - pano_left

    pano_top = int(min(top_left[1], top_right[1], 0))
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    H = pano_bottom - pano_top

    size = (W, H)

    if DEBUG:
        print 'Panodimensions'
        print pano_top
        print pano_bottom

    # offset of first image relative to panorama
    X = int(min(top_left[0], bottom_left[0], 0))
    Y = int(min(top_left[1], top_right[1], 0))
    if X < 0:
    	X = -X
    else :
    	X = 0

    if Y < 0:
    	Y = -Y
    else :
    	Y = 0
    offset = (X, Y)

    if DEBUG:
        print 'Calculated size:'
        print size
        print 'Calculated offset:'
        print offset

    return (size, offset)

def crop(final_img):

	final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
	_,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	max_area = 0
	best_rect = (0,0,0,0)
	    
	for cnt in contours:
		# print cv2.boundingRect(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		# print "Bounding Rectangle: ", (x,y,w,h)
    
		deltaHeight = h-y
		deltaWidth = w-x
    
		area = deltaHeight * deltaWidth
    
		if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
			max_area = area
			best_rect = (x,y,w,h)
    
	if ( max_area > 0 ):
		final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
			best_rect[0]:best_rect[0]+best_rect[2]]
    
		# utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
		# cv2.destroyAllWindows()
    
		final_img = final_img_crop

	return final_img

## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):
    ## TODO: Combine the two images into one.
    ## TODO: (Overwrite the following 5 lines with your answer.)
	base_img_rgb = image1
	next_img = image2
	H_inv = np.linalg.inv(homography)
	min_x, min_y, max_x, max_y = findDimensions(image2, H_inv)

	max_x = max(max_x, base_img_rgb.shape[1])
	max_y = max(max_y, base_img_rgb.shape[0])

	move_h = np.matrix(np.identity(3), np.float32)
	if ( min_x < 0 ):
		move_h[0,2] += -min_x
		max_x += -min_x
    
	if ( min_y < 0 ):
		move_h[1,2] += -min_y
		max_y += -min_y

	ox = int(move_h[0,2])
	oy = int(move_h[1,2])

	mod_inv_h = move_h * H_inv

	img_w = int(math.ceil(max_x))
	img_h = int(math.ceil(max_y))

	# return final_img
	(h1, w1) = image1.shape[:2]

	# # draw the transformed image2
	panorama = cv2.warpPerspective(np.array(image2,dtype=np.uint8), mod_inv_h, (img_w, img_h))

	panorama[oy:h1+oy, 0:w1] = image1


	return crop(np.array(panorama, dtype=np.uint8))