import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec

# Define the directories for the Vanilla GAN and MMD GAN
vanilla_gan_dir = 'results/3-final_tests/a_vanilla_gan_1d/0-tests/'
mmd_gan_dir = 'results/3-final_tests/b_mmd_gan_1d/0-tests/'
image_name = 'histogram_old.png'

# Define the subdirectories for the different distributions
distributions = ['gaussian', 'pareto', 'uniform', 'cauchy']

# Create the figure and the subplots
fig, axs = plt.subplots(2, len(distributions) + 1, figsize=(len(distributions)*10, 10))

# Set the row titles in the first column, and hide these subplots
for j in range(2):
    axs[j, 0].text(0.65, 0.5, ['Vanilla GAN', 'MMD GAN'][j], ha='left', va='center', fontsize=16)
    axs[j, 0].axis('off')

# Iterate over the distributions
for i, dist in enumerate(distributions):
    # Adjust index for the images since we added an additional column
    i += 1

    # Get the list of subdirectories in the distribution directory for Vanilla GAN and MMD GAN
    vanilla_gan_subdirs = next(os.walk(os.path.join(vanilla_gan_dir, dist)))[1]
    mmd_gan_subdirs = next(os.walk(os.path.join(mmd_gan_dir, dist)))[1]
    
    # Find the subdirectory that starts with the distribution name
    vanilla_gan_subdir = next((s for s in vanilla_gan_subdirs if s.startswith(dist)), None)
    mmd_gan_subdir = next((s for s in mmd_gan_subdirs if s.startswith(dist)), None)
    
    # If we found the subdirectories
    if vanilla_gan_subdir is not None and mmd_gan_subdir is not None:
        # Create the filepaths for the Vanilla GAN and MMD GAN
        vanilla_gan_filepath = os.path.join(vanilla_gan_dir, dist, vanilla_gan_subdir, image_name)
        mmd_gan_filepath = os.path.join(mmd_gan_dir, dist, mmd_gan_subdir, image_name)

        # Read the images
        vanilla_gan_img = mpimg.imread(vanilla_gan_filepath)
        mmd_gan_img = mpimg.imread(mmd_gan_filepath)

        # Display the images
        axs[0, i].imshow(vanilla_gan_img)
        axs[1, i].imshow(mmd_gan_img)

        # Remove the axis
        axs[0, i].axis('off')
        axs[1, i].axis('off')

        # Set the column title
        axs[0, i].text(0.5, 1.1, dist.capitalize(), ha="center", va="center", 
            transform=axs[0, i].transAxes, fontsize=14)

# Reduce space between plots and create more space on the left for labels
plt.subplots_adjust(wspace=0, hspace=0.02, left=0.30)

# Save the figure
plt.savefig('results/3-final_tests/a_vanilla_gan_1d/0-tests/combined_plots.png', bbox_inches='tight', dpi=300)

from PIL import Image

# Open the image file
img = Image.open('results/3-final_tests/a_vanilla_gan_1d/0-tests/combined_plots.png')
width, height = img.size

# Set the dimensions of the crop box
left = width * 0.12
top = 0
right = width
bottom = height

# Crop the image
img_cropped = img.crop((left, top, right, bottom))

# Save the cropped image
img_cropped.save('results/3-final_tests/a_vanilla_gan_1d/0-tests/combined_plots.png')
