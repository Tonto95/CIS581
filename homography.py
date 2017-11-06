import cv2

def homographize(src, dest):
    return cv2.findHomography(src, dest, cv2.RANSAC,5.0)