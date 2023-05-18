import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
import json

# Define the directories for the Vanilla GAN and MMD GAN
directory = 'different_distributions'
image_name = 'histogram_pdf.png'
save_dir = 'results/4-table_plots/'
mode = "gan_vs_qaaan_mmd"
if mode == "e_vanilla_qaaan" or mode == "f_mmd_qaaan":
    gan_name = mode
    dir_1 = 'results/3-final_tests/'+gan_name+'/classical/' + directory + '/'
    dir_2 = 'results/3-final_tests/'+gan_name+'/quantum/' + directory + '/'
    if gan_name == 'e_vanilla_qaaan':
        file_name = '2-classical_vs_quantum_vanilla_qaaan'
        column_titles = ["Classical Vanilla QAAAN", "Quantum Vanilla QAAAN"]
    elif gan_name == 'f_mmd_qaaan':
        file_name = '3-classical_vs_quantum_mmd_qaaan'
        column_titles = ["Classical MMD QAAAN", "Quantum MMD QAAAN"]
elif mode == "gan":
    dir_1 = 'results/3-final_tests/a_vanilla_gan_1d/' + directory + '/'
    dir_2 = 'results/3-final_tests/e_vanilla_qaaan/quantum/' + directory + '/'
    file_name = '1-vanilla_vs_mmd_gan'
    column_titles = ["Vanilla GAN", "MMD GAN"]
elif mode == "gan_vs_qaaan_vanilla":
    dir_1 = 'results/3-final_tests/a_vanilla_gan_1d/2-epochs=900/' + directory + '/'
    dir_2 = 'results/3-final_tests/e_vanilla_qaaan/quantum/' + directory + '/'
    save_dir = 'results/4-table_plots/'
    file_name = '4-vanilla_gan_vs_qaaan'
    column_titles = ["Vanilla GAN", "Vanilla QAAAN"]
elif mode == "gan_vs_qaaan_mmd":
    dir_1 = 'results/3-final_tests/b_mmd_gan_1d/2-epochs=900/' + directory + '/'
    dir_2 = 'results/3-final_tests/f_mmd_qaaan/quantum/' + directory + '/'
    save_dir = 'results/4-table_plots/'
    file_name = '5-mmd_gan_vs_qaaan'
    column_titles = ["MMD GAN", "MMD QAAAN"]


# Define the subdirectories for the different distributions
distributions = ['uniform', 'gaussian', 'cauchy', 'pareto']

# Create the figure and the subplots
fig, axs = plt.subplots(len(distributions), 2, figsize=(10, len(distributions)*5))

# Set the column titles
axs[0, 0].set_title(column_titles[0], fontsize=12)
axs[0, 1].set_title(column_titles[1], fontsize=12)

latex_table_rows = []

# Iterate over the distributions
for i, dist in enumerate(distributions):
    # Get the list of subdirectories in the distribution directory for Vanilla GAN and MMD GAN
    vanilla_gan_subdirs = next(os.walk(os.path.join(dir_1, dist)))[1]
    mmd_gan_subdirs = next(os.walk(os.path.join(dir_2, dist)))[1]
    
    # Find the subdirectory that starts with the distribution name
    vanilla_gan_subdir = next((s for s in vanilla_gan_subdirs if s.startswith(dist)), None)
    mmd_gan_subdir = next((s for s in mmd_gan_subdirs if s.startswith(dist)), None)
    
    # If we found the subdirectories
    if vanilla_gan_subdir is not None and mmd_gan_subdir is not None:
        # Create the filepaths for the Vanilla GAN and MMD GAN
        vanilla_gan_filepath = os.path.join(dir_1, dist, vanilla_gan_subdir, image_name)
        mmd_gan_filepath = os.path.join(dir_2, dist, mmd_gan_subdir, image_name)

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
        

        ### Wasserstein Distances table creation
        # Create the filepaths for the parameters.json files
        vanilla_gan_params_filepath = os.path.join(dir_1, dist, vanilla_gan_subdir, "parameters.json")
        mmd_gan_params_filepath = os.path.join(dir_2, dist, mmd_gan_subdir, "parameters.json")

        # Load the parameters.json files
        with open(vanilla_gan_params_filepath, 'r') as f:
            vanilla_gan_params = json.load(f)
        with open(mmd_gan_params_filepath, 'r') as f:
            mmd_gan_params = json.load(f)

        # Get the "wasserstein_distance" parameter
        vanilla_gan_wasserstein_distance = vanilla_gan_params.get("wasserstein_distance", "N/A")
        mmd_gan_wasserstein_distance = mmd_gan_params.get("wasserstein_distance", "N/A")

        # Add a row to the LaTeX table
        latex_table_rows.append(f"{dist.capitalize()} & {vanilla_gan_wasserstein_distance} & {mmd_gan_wasserstein_distance} \\\\")



# Reduce space between plots
plt.subplots_adjust(wspace=0.02, hspace=-0.5)

# Save the figure
plt.savefig(save_dir+file_name+'.png', bbox_inches='tight', dpi=600)


# Combine the rows into a string for the LaTeX table
latex_table = "\\begin{tabular}{|c|c|c|}\n\\hline\n"
latex_table += "& \multicolumn{2}{c|}{Wasserstein Distance} \\\\\n\\cline{2-3}\n"
latex_table += "Distribution & Vanilla GAN & MMD GAN \\\\\n\\hline\n"
latex_table += "\n".join(latex_table_rows)
latex_table += "\n\\hline\n\\end{tabular}"

# Write the LaTeX table to a .txt file
with open(save_dir+file_name+'.txt', 'w') as f:
    f.write(latex_table)