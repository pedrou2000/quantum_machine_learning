import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json

class GAN2D:
    def __init__(self, latent_dim, epochs, batch_size, target_mean, target_sd, noise_low, noise_high, results_path, 
                 num_batches_per_epoch):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_mean = target_mean
        self.target_sd = target_sd
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.results_path = results_path
        self.num_batches_per_epoch = num_batches_per_epoch
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.create_gan()
        self.compile_models()
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []

    def target_distribution(self):
        return np.random.multivariate_normal(self.target_mean, self.target_sd, self.batch_size)

    def noise_distribution(self):
        return np.random.uniform(self.noise_low, self.noise_high, (self.batch_size, self.latent_dim))

    def create_generator(self):
        model = keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_dim=self.latent_dim))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        return model

    def create_discriminator(self):
        model = keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(2,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def create_gan(self):
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def compile_models(self):
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        self.discriminator.trainable = False
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self):
        real_labels = np.ones((self.batch_size, 1))

        for epoch in range(self.epochs):
            for batch in range(self.num_batches_per_epoch):
                # Train discriminator on real data
                real_data = self.target_distribution()
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                noise = self.noise_distribution()
                generated_data = self.generator.predict(noise, verbose=0)
                fake_labels = np.zeros((self.batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

                # Train generator
                noise = self.noise_distribution()
                g_loss = self.gan.train_on_batch(noise, real_labels)

            # Save losses
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real}, D_loss_fake: {d_loss_fake}, G_loss: {g_loss}")
                self.d_losses_real.append(d_loss_real)
                self.d_losses_fake.append(d_loss_fake)
                self.g_losses.append(g_loss)


    def plot_losses(self, folder_path):
        plt.plot(self.d_losses_real, label='D_loss_real')
        plt.plot(self.d_losses_fake, label='D_loss_fake')
        plt.plot(self.g_losses, label='G_loss')
        plt.xlabel('Epoch (x100)')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses')
        plt.ylim(0, 2)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def plot_results(self, folder_path):
        noise = self.noise_distribution()
        generated_data = self.generator.predict(noise, verbose=0)
        real_data = self.target_distribution()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        hist, xedges, yedges = np.histogram2d(generated_data[:, 0], generated_data[:, 1], bins=20)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.8 * (xedges[1] - xedges[0])
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='r', shade=True, label='Generated Data', alpha=0.6)

        hist, xedges, yedges = np.histogram2d(real_data[:, 0], real_data[:, 1], bins=20)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.8 * (xedges[1] - xedges[0])
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='b', shade=True, label='Real Data', alpha=0.6)

        # Add legend manually, as ax.bar3d doesn't support the 'label' parameter
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='r', lw=4),
                        Line2D([0], [0], color='b', lw=4)]
        ax.legend(custom_lines, ['Generated Data', 'Real Data'])

        plt.savefig(f'{folder_path}3dhistogram.png', dpi=300)
        plt.close()


    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'target_mean': self.target_mean.tolist(),  # Convert to list
            'target_sd': self.target_sd.tolist(),  # Convert to list
            'noise_low': self.noise_low,
            'noise_high': self.noise_high,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)

    
    def create_result_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = self.results_path + f'2dgaussian_{timestamp}/'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def plot_and_save(self):
        folder_path = self.create_result_folder()
        self.plot_results(folder_path)
        self.plot_losses(folder_path)
        self.save_parameters_to_json(folder_path)

if __name__ == "__main__":
    # Parameters
    epochs=10000
    num_batches_per_epoch = 3
    batch_size=1000
    latent_dim=100
    noise_low=-1
    noise_high=1
    results_path='results/gan/2d/'

    target_mean = np.array([0, 0])
    target_sd = np.array([[0.5, 0], [0, 0.5]])

    print(f"Running GAN for 2D Gaussian distribution with mean: {target_mean}, sd: {target_sd}")
    gan = GAN2D(latent_dim=latent_dim, epochs=epochs, batch_size=batch_size,
                target_mean=target_mean, target_sd=target_sd, noise_low=noise_low, 
                noise_high=noise_high, results_path=results_path, num_batches_per_epoch=num_batches_per_epoch)

    gan.train()
    gan.plot_and_save()
