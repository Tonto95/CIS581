# 
# Offsetting 
# the key: http://stackoverflow.com/questions/6087241/opencv-warpperspective
#

# For the ocean panorama, SIFT found a lot more features. This 
# resulted in a much better stitching. (SURF only found 4 and it
# warped considerably)

# Test cases
# python stitch.py Image1.jpg Image2.jpg -a SIFT
# python stitch.py Image2.jpg Image1.jpg -a SIFT
# python stitch.py ../stitcher/images/image_5.png ../stitcher/images/image_6.png -a SIFT
# python stitch.py ../stitcher/images/image_6.png ../stitcher/images/image_5.png -a SIFT
# python stitch.py ../vashon/01.JPG ../vashon/02.JPG -a SIFT
# python stitch.py panorama_vashon2.jpg ../vashon/04.JPG -a SIFT
# python stitch.py ../books/02.JPG ../books/03.JPG -a SIFT

# coding: utf-8
import cv2, numpy as np
import math
import argparse as ap
from main import master
from find_correspondances import extract_features

DEBUG = False
MIN_MATCHES_COUNT = 4


## 2. Find corresponding features between the images. [2] ----------------
def find_correspondences(kp1, des1, kp2, des2):

  # Initialize FLANN algorithm parameters to be used in the feature matching process 
  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks=50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)

  if len(matches) > MIN_MATCHES_COUNT:
    points1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

  
  ## TODO: Look up corresponding keypoints.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  # points1 = np.array([k.pt for k in keypoints1], np.float32)
  # points2 = np.array([k.pt for k in keypoints1], np.float32)

  return (points1, points2)


## 3. Calculate the size and offset of the stitched panorama. [5] --------



def calculate_size(size_image1, size_image2, homography):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  # h1, w1,_) = image1.shape
  # (h2, w2,_) = image2.shape
  
  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  
  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  
  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)

  return (size, offset)


## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  ## TODO: Combine the two images into one.
  ## TODO: (Overwrite the following 5 lines with your answer.)
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  ox = offset[0]
  oy = offset[1]
  # (ox, oy) = offset[]
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  if DEBUG:
    print homography
  homography = homography * translation
  # print homography
  
  # draw the transformed image2
  panorma = cv2.warpPerspective(image2, homography, size)
  
  panorama[oy:h1+oy, ox:ox+w1] = image1  
  # panorama[:h1, :w1] = image1  

  ## TODO: Draw the common feature keypoints.

  return panorama

def merge_images_translation(image1, image2, offset):

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  (ox, oy) = offset
  ox = int(ox)
  oy = int(oy)
  oy = 0
  
  image = np.zeros((h1+oy, w1+ox, 3), np.uint8)
  
  image[:h1, :w1] = image1
  image[:h2, ox:ox+w2] = image2
  
  return image



if __name__ == "__main__":
  path = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/test_imgs"
  out_path = "/Users/Ahmed/Documents/CIS 581/CIS581Project3/Mini Project/CIS581/panorama.jpg"
  size = np.array([1632,1224,3])

  pics = master(path, size)

  ## Load images.
  image1 = pics[0]
  image2 = pics[15]

  image1 = np.array(image1, np.uint8)
  image2 = np.array(image2, np.uint8)

  ## Detect features and compute descriptors.
  (descriptors1, keypoints1) = extract_features(image1)
  (descriptors2, keypoints2) = extract_features(image2)
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  ## Find corresponding features.
  (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  print len(points1), "features matched"
  
  ## Visualise corresponding features.
  # correspondences = draw_correspondences(image1, image2, points1, points2)
  # cv2.imwrite("out/correspondences.jpg", correspondences)
  # print 'Wrote correspondences.jpg'
  
  ## Find homography between the views.
  (homography, _) = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
  # homography = find_H(points1, points2)
  # print homography
  
  ## Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  print "offset: %ix%i" % offset 
  print "output size: %ix%i" % size
  
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  cv2.imwrite(out_path, panorama)
  print 'Wrote panorama.jpg'