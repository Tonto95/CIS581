
# coding: utf-8
import cv2, numpy as np
import math
import argparse as ap
import image_loader

DEBUG = False



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
    offset = (-X, -Y)

    if DEBUG:
        print 'Calculated size:'
        print size
        print 'Calculated offset:'
        print offset

    return (size, offset)


## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):
    ## TODO: Combine the two images into one.
    ## TODO: (Overwrite the following 5 lines with your answer.)
    (h1, w1) = image1.shape[:2]

    (ox, oy) = offset

    translation = np.matrix([
        [1.0, 0.0, ox],
        [0, 1.0, oy],
        [0.0, 0.0, 1.0]
    ])

    if DEBUG:
        print homography
    homography = translation * homography
    # print homography

    # draw the transformed image2
    panorama = cv2.warpPerspective(image2, homography, size)

    panorama[oy:h1 + oy, ox:ox + w1] = image1

    return panorama