import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker = cv2.aruco.generateImageMarker(dictionary, 0, 500)

# # Add white border
# marker = cv2.copyMakeBorder(
#     marker,
#     80, 80, 80, 80,
#     cv2.BORDER_CONSTANT,
#     value=255
# )

# Flip horizontally to cancel MuJoCo mirroring
marker = cv2.flip(marker, 1)

cv2.imwrite("integration/assets/aruco_0.png", marker)
print("Saved flipped marker")