import matplotlib.pyplot as plt
import glob
import tensorflow_docs.vis.embed as embed
import imageio.v2 as imageio

def plot_distributions(dist_1, dist_2, color_dist_1, color_dist_2, images_path, epoch, n_bins = 10):
    plt.hist(x=dist_1, bins=n_bins, color=color_dist_1, alpha=0.7, rwidth=0.85)
    plt.hist(x=dist_2, bins=n_bins, color=color_dist_2, alpha=0.7, rwidth=0.85)
    plt.savefig(images_path + 'image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

def create_gif(images_path = '2-gan/results/images/', gif_file = '2-gan/results/gan.gif'):
    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = glob.glob(images_path + 'image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    
    embed.embed_file(gif_file)
