import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cv2
from machinevisiontoolbox import CentralCamera, Image, FileCollection, mkcube, mkgrid
from spatialmath import SE3

from sensing.aruco_camera import create_camera

create_camera(lens_size=0.012, pixel_size=10e-6, image_x=1024, image_y=2048, name=cam2)