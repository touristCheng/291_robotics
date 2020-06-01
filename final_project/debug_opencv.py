import numpy as np
import matplotlib.pyplot as plt


import cv2

rect1 = np.array([[-1, 0], [1, 0], [-1, -1], [1, -1]])
rect2 = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
rbox = cv2.minAreaRect(rect2)
center, size, rot = rbox


print(center, rot, size)