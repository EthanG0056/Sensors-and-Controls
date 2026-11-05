import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker = cv2.aruco.generateImageMarker(dictionary, 1, 400)  # Change for each aruco marker (0 = [0, 500], 1 = [1, 400], 2 = [3, 300])

# Flip horizontally to cancel MuJoCo mirroring
marker = cv2.flip(marker, 1)

cv2.imwrite("integration/assets/aruco_1.png", marker)       # Change for each aruco marker

print("Saved flipped marker")