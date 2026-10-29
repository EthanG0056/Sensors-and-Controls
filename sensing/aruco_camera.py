import numpy as np
import matplotlib.pyplot as plt
import cv2
from machinevisiontoolbox import CentralCamera, Image, FileCollection, mkcube, mkgrid
from spatialmath import SE3

np.set_printoptions(precision=3, suppress=True)
plt.rcParams['figure.figsize'] = (7, 5)
print('toolbox loaded')

cam = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=[640, 512], name='lab camera')
print(cam)
print('K =')
print(cam.K)
print('horizontal, vertical field of view (deg):', np.degrees(cam.fov()))

P = np.array([0.3, 0.2, 4.0])       # a world point, 4 m in front of the camera
p = cam.project_point(P)
print('world point', P, ' -> pixel', p.flatten())

ray = np.array([0.1, 0.05, 1.0])                 # direction
for depth in [2.0, 5.0, 20.0]:
    P = ray * depth
    print(f'P = {P},  pixel = {cam.project_point(P).flatten()}')

X, Y, Z = mkcube(0.4, pose=SE3(0, 0, 2), edge=True)   # 0.4 m cube, 2 m in front
plt.figure()
cam.plot_wireframe(X, Y, Z, color='b')
plt.title('cube projected onto the image plane')
plt.show()

