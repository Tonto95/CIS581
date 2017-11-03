import cv2

def warp(img, H):
    result = cv2.warpPerspective(img, (H, img.shape[0] * 2, img.shape[1]))
