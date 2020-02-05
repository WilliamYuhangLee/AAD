import Augmentor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

## A quick walkthrough of Augmentor's features

## Minimalist usage example

# Create a pipeline p, specifies path to the pictures to be augmented
p = Augmentor.Pipeline("data")

# If desired, ground truth can be augmented in parallel; files with same names will be augmented together
# p.ground_truth("path_to_ground_truth")

# Add operators to the pipeline 'p'
p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)  # probability = chance of applying this operator

# Sample a specified number of pictures from augmented data, output will be stored in a separate directory
p.sample(4, multi_threaded=False)  # supports multi-threading

## Multiple Mask Image Augmentation
image1 = Image.open("data/grid.png")
image2 = Image.open("data/8.png")
images = [[np.asarray(image1)], [np.asarray(image2)]]  # images must be in List of Lists [[]] format
p = Augmentor.DataPipeline(images, labels=None)
p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
augmented_images = p.sample(1)
plt.imshow(augmented_images[0][0])
plt.show()

## Supported operators

# Rotations
p.rotate(probability=1, max_left_rotation=0, max_right_rotation=0)
p.rotate90(probability=1)
p.rotate180(probability=1)
p.rotate270(probability=1)
p.rotate_random_90(probability=1)
p.rotate_without_crop(probability=1, max_left_rotation=0, max_right_rotation=0, expand=False, fillcolor=None)

# Flipping
p.flip_left_right(probability=1)
p.flip_top_bottom(probability=1)
p.flip_random(probability=1)

# Resize/cropping
p.resize(probability=1, width=0, height=0, resample_filter="BICUBIC")
p.crop_by_size(probability=1, width=0, height=0, centre=True)
p.crop_centre(probability=1, percentage_area=0, randomise_percentage_area=False)
p.crop_random(probability=1, percentage_area=0, randomise_percentage_area=False)

# Random transformations
p.random_brightness(probability=1, min_factor=0, max_factor=0)
p.random_color(probability=1, min_factor=0, max_factor=0)
p.random_contrast(probability=1, min_factor=0, max_factor=0)
p.random_distortion(probability=1, grid_width=0, grid_height=0, magnitude=0)
p.random_erasing(probability=1, rectangle_area=0.1)

# Color manipulation
p.black_and_white(probability=1, threshold=128)
p.greyscale(probability=1)

# Skewing
p.skew(probability=1, magnitude=1)
p.shear(probability=1, max_shear_left=25, max_shear_right=25)
