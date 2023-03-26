import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json

class GAN:
    def __init__(self, latent_dim, epochs, batch_size, target_mean, target_sd, target_normal, noise_low, noise_high, results_path):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_mean = target_mean
        self.target_sd = target_sd
        self.target_normal = target_normal
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.results_path = results_path
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.create_gan()
        self.compile_models()
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []

    def target_distribution(self):
        if self.target_normal:
            return np.random.normal(self.target_mean, self.target_sd, (self.batch_size, self.batch_size))
        else:
            return np.random.uniform(self.target_mean, self.target_sd, (self.batch_size, self.batch_size))

    def noise_distribution(self):
        return np.random.uniform(self.noise_low, self.noise_high, (self.batch_size, self.latent_dim))

    def create_generator(self):
        model = keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_dim=self.latent_dim))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.batch_size, activation='linear'))
        return model

    def create_discriminator(self):
        model = keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.batch_size,)))
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
        real_data = self.target_distribution()
        real_labels = np.ones((self.batch_size, 1))

        for epoch in range(self.epochs):
            # Train discriminator on real data
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
        generated_data = self.generator.predict(noise, verbose=0).flatten()

        target_data = self.target_distribution().flatten()
        
        plt.hist(generated_data, bins=50, alpha=0.6, label='Generated Data')
        plt.hist(target_data, bins=50, alpha=0.6, label='Target Distribution')
        plt.legend()

        plt.savefig(f'{folder_path}histogram.png', dpi=300)
        plt.close()


    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'target_mean': self.target_mean,
            'target_sd': self.target_sd,
            'target_normal': self.target_normal,
            'noise_low': self.noise_low,
            'noise_high': self.noise_high,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)

    def create_result_folder(self):
        distribution_type = 'normal' if self.target_normal else 'uniform'
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = self.results_path + f'{distribution_type}_{self.target_mean}_{self.target_sd}_{timestamp}/'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def plot_and_save(self):
        folder_path = self.create_result_folder()
        self.plot_results(folder_path)
        self.plot_losses(folder_path)
        self.save_parameters_to_json(folder_path)








if __name__ == "__main__":
    # Parameters
    epochs=1000
    latent_dim=100
    batch_size=128
    noise_low=-1
    noise_high=1
    results_path='results/gan/1d/setwise/'
    """

    # Normal distributionss
    means = [0, 5, 10]
    sds = [1, 1.5, 2, 3]

    for mean in means:
        for sd in sds:
            print(f"Running GAN for normal distribution with mean: {mean}, sd: {sd}")
            gan = GAN(latent_dim=latent_dim, epochs=epochs, batch_size=batch_size,
                      target_mean=mean, target_sd=sd, target_normal=True,
                      noise_low=noise_low, noise_high=noise_high, results_path=results_path)

            gan.train()
            gan.plot_and_save()

    """
    # Uniform distributions
    # lows = np.linspace(-30, 30, 5)
    # highs = np.linspace(-30, 30, 5)
    #edges = [(-1,1), (0, 0.1), (0,2), (10,11), (10,15)]
    edges = [(0, 2)]

    for (low, high) in edges:
        print(f"Running GAN for uniform distribution with low: {low}, high: {high}")
        gan = GAN(latent_dim=latent_dim, epochs=epochs, batch_size=batch_size,
                    target_mean=low, target_sd=high, target_normal=False,
                    noise_low=noise_low, noise_high=noise_high, results_path=results_path)

        gan.train()
        gan.plot_and_save()

