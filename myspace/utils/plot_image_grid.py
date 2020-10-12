import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import numpy as np
import cv2

image_file = "/home/sdb/wangshentao/myspace/thesis/data/VisDrone2019-MOT-train/sequences/uav0000138_00000_v/0000001.jpg"
image = cv2.imread(image_file)
image = cv2.resize(image, (1080, 608))
image = image.astype(np.uint8)
image = image[250:450, 400:600]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
minorLocator = MultipleLocator(4)

fig = plt.figure()
ax = plt.axes()
ax.imshow(image)
ax.yaxis.set_minor_locator(minorLocator)
ax.xaxis.set_minor_locator(minorLocator)
ax.grid(which='minor')

plt.xticks([])
plt.yticks([])
plt.show()
