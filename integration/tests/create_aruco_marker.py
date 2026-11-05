import cv2


dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker = cv2.aruco.generateImageMarker(
    dictionary,
    0,
    500
)

cv2.imwrite(
    "integration/assets/aruco_0.png",
    marker
)

print("ArUco marker created")