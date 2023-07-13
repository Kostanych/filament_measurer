import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage import measure
from skimage.draw import polygon2mask
from skimage.io import imread
from skimage.morphology import convex_hull_image
from skimage.util import invert


# Construct some test data
full_path = "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\input\\photo_4.jpg"
image = imread(full_path, as_gray=True)
image[:, 0] = 1
image[:, -1] = 1

# Find contours at a constant value of 0.8
contours = measure.find_contours(image, 0.9)

# Find largest contour
largest_contour = max(contours, key=lambda contour: len(contour))

# Make mask
mask = polygon2mask(image.shape, largest_contour)
mask = invert(convex_hull_image(mask))
mask = mask.astype(float)

print(mask)
print(mask.shape)

# Display the image and plot the largest contour
fig, ax = plt.subplots()
ax.imshow(mask, cmap=plt.cm.gray)

ax.plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
