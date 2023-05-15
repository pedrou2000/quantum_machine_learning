import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

# Define the directories for the Vanilla GAN and MMD GAN
directory = 'different_distributions'
image_name = 'histogram_pdf.png'
gan_name = 'e_vanilla_qaaan'
vanilla_gan_dir = 'results/3-final_tests/'+gan_name+'/classical/' + directory + '/'
mmd_gan_dir = 'results/3-final_tests/'+gan_name+'/quantum/' + directory + '/'
save_dir = 'results/3-final_tests/'+gan_name+'/'
save_dir += 'classical_vs_quantum_'+gan_name+'.png'

# Define the subdirectories for the different distributions
distributions = ['uniform', 'gaussian', 'cauchy', 'pareto']

# Create the figure and the subplots
fig, axs = plt.subplots(len(distributions), 2, figsize=(10, len(distributions)*5))

# Set the column titles
axs[0, 0].set_title('Classical Vanilla QAAAN', fontsize=12)
axs[0, 1].set_title('Quantum Vanilla QAAAN', fontsize=12)

# Iterate over the distributions
for i, dist in enumerate(distributions):
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

        # Read the images with PIL's Image.open for more manipulation options
        vanilla_gan_img = Image.open(vanilla_gan_filepath)
        mmd_gan_img = Image.open(mmd_gan_filepath)

        # Get the size of the images
        width, height = vanilla_gan_img.size

        # Define the left, top, right, and bottom coordinates for cropping
        left = width * 0.033  # 5% of the width
        top = 0
        right = width
        bottom = height * 0.97

        # Crop the images
        vanilla_gan_img_cropped = vanilla_gan_img.crop((left, top, right, bottom))
        mmd_gan_img_cropped = mmd_gan_img.crop((left, top, right, bottom))

        # Convert the PIL Images back to NumPy arrays for matplotlib
        vanilla_gan_img_cropped = np.array(vanilla_gan_img_cropped)
        mmd_gan_img_cropped = np.array(mmd_gan_img_cropped)

        # Display the images
        axs[i, 0].imshow(vanilla_gan_img_cropped)
        axs[i, 1].imshow(mmd_gan_img_cropped)

        # Remove the axis
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

        # Set the row title# Set the row title
        axs[i, 0].text(-0.3,0.5, dist.capitalize(), ha="left", va="center", 
                    transform=axs[i, 0].transAxes, fontsize=12)


# Reduce space between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.5)

# Save the figure
plt.savefig(save_dir, bbox_inches='tight', dpi=600)
