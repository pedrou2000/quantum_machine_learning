import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json
from scipy.stats import norm, uniform, cauchy, pareto
from keras.optimizers import Adam

class GAN:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.latent_dim = hyperparameters['network']['latent_dim']
        self.epochs = hyperparameters['training']['epochs']
        self.batch_size = hyperparameters['training']['batch_size']
        self.save_frequency = hyperparameters['training']['save_frequency']
        self.update_ratio_critic = hyperparameters['training']['update_ratio_critic']
        self.learning_rate = hyperparameters['training']['learning_rate']
        self.mean = hyperparameters['distributions']['mean']
        self.variance = hyperparameters['distributions']['variance']
        self.target_dist = hyperparameters['distributions']['target_dist']
        self.input_dist = hyperparameters['distributions']['input_dist']
        self.plot_size = hyperparameters['plotting']['plot_size']
        self.n_bins = hyperparameters['plotting']['n_bins']
        self.results_path = hyperparameters['plotting']['results_path']
        self.layers_gen = hyperparameters['network']['layers_gen']
        self.layers_disc = hyperparameters['network']['layers_disc']
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.create_gan()
        self.compile_models()
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []

    def sample_real_data(self):
        if self.target_dist == "gaussian":
            return np.random.normal(self.mean, self.variance, self.batch_size)
        elif self.target_dist == "uniform":
            return np.random.uniform(self.mean, self.variance, self.batch_size)
        elif self.target_dist == "cauchy":
            return cauchy.rvs(self.mean, self.variance, self.batch_size)
        elif self.target_dist == "pareto":
            return pareto.rvs(self.mean, self.variance, self.batch_size)

    def sample_noise(self, plot=False):
        if not plot: 
            if self.input_dist == "gaussian":
                return np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            elif self.input_dist == "uniform":
                return np.random.uniform(0, 1,(self.batch_size, self.latent_dim))
        else:
            if self.input_dist == "gaussian":
                return np.random.normal(0, 1, (self.plot_size, self.latent_dim))
            elif self.input_dist == "uniform":
                return np.random.uniform(0, 1,(self.plot_size, self.latent_dim))

    def create_generator(self):
        model = keras.Sequential()
        
        model.add(layers.Dense(self.layers_gen[0], activation='relu', input_dim=self.latent_dim))
        for n_layer in self.layers_gen[1:-1]:
            model.add(layers.Dense(n_layer, activation='relu'))
        model.add(layers.Dense(self.layers_gen[-1], activation='linear'))
        return model

    def create_discriminator(self):
        model = keras.Sequential()

        model.add(layers.Dense(self.layers_disc[0], activation='relu', input_shape=(1,)))
        for n_layer in self.layers_disc[1:-1]:
            model.add(layers.Dense(n_layer, activation='relu'))
        model.add(layers.Dense(self.layers_disc[len(self.layers_disc)-1], activation='sigmoid'))
        return model

    def create_gan(self):
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def compile_models(self):
        custom_adam = Adam(learning_rate=self.learning_rate)
        self.discriminator.compile(optimizer=custom_adam, loss='binary_crossentropy')
        self.discriminator.trainable = False
        self.gan.compile(optimizer=custom_adam, loss='binary_crossentropy')

    def train(self):
        real_data = self.sample_real_data()
        real_labels = np.ones((self.batch_size, 1))

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0

        for epoch in range(self.epochs):
            for _ in range(self.update_ratio_critic):
                # Train discriminator on real data
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                noise = self.sample_noise()
                generated_data = self.generator.predict(noise, verbose=0)
                fake_labels = np.zeros((self.batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # Train generator
            noise = self.sample_noise()
            g_loss = self.gan.train_on_batch(noise, real_labels)

            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            g_loss_total += g_loss

            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real_total}, D_loss_fake: {d_loss_fake_total}, G_loss: {g_loss_total}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0


    def plot_results(self, folder_path):
        noise = self.sample_noise(plot=True)
        generated_data = self.generator.predict(noise, verbose=0).flatten()

        plt.hist(generated_data, bins=self.n_bins, alpha=0.6, label='Generated Data', density=True)

        # Plot PDFs
        x_values = np.linspace(self.mean - 4 * np.sqrt(self.variance), self.mean + 4 * np.sqrt(self.variance), 1000)

        if self.target_dist == "gaussian":
            pdf_values = norm.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "uniform":
            lower_bound = self.mean
            upper_bound = self.variance
            scale = upper_bound - lower_bound
            pdf_values = uniform.pdf(x_values, loc=self.mean, scale=scale)
        elif self.target_dist == "cauchy":
            pdf_values = cauchy.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "pareto":
            pdf_values = pareto.pdf(x_values, b=self.mean, scale=np.sqrt(self.variance))

        pdf_values = pdf_values / (pdf_values.sum() * np.diff(x_values)[0])  # normalize the PDF

        plt.plot(x_values, pdf_values, label="Real PDF")

        plt.legend()

        plt.savefig(f'{folder_path}histogram.png', dpi=300)
        plt.close()

    def plot_losses(self, folder_path):
        plt.plot(self.d_losses_real, label='D_loss_real')
        plt.plot(self.d_losses_fake, label='D_loss_fake')
        plt.plot(self.g_losses, label='G_loss')
        plt.xlabel('Epoch x '+str(self.save_frequency))
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses')
        plt.ylim(0, 2)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': self.hyperparameters,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
    
    def create_result_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = self.results_path + f'{self.target_dist}_{self.mean}_{self.variance}_{timestamp}/'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def plot_and_save(self):
        folder_path = self.create_result_folder()
        self.plot_results(folder_path)
        self.plot_losses(folder_path)
        self.save_parameters_to_json(folder_path)


def simple_main(hyperparameters):
    gan = GAN(hyperparameters)
    gan.train()
    gan.plot_and_save()

def complex_main(hyperparameters):
    update_ratio_critics = [1,2,3,4,5]

    for update_ratio_critic in update_ratio_critics:
        hyperparameters['training']['update_ratio_critic'] = update_ratio_critic
        
        gan = GAN(hyperparameters)
        gan.train()
        gan.plot_and_save()



if __name__ == "__main__":
    one_run = True

    hyperparameters = {
        'training': {
            'epochs': 100,
            'batch_size': 128,
            'save_frequency': 1,
            'update_ratio_critic': 5,
            'learning_rate': 0.001,
        },
        'network': {
            'latent_dim': 100,
            'layers_gen': [2, 13, 7, 1],
            'layers_disc': [11, 29, 11, 1],
        },
        'distributions': {
            'mean': 5,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            #'results_path': 'results/2-tests/1-gan/1-vanilla_gan_1d/0-tests/',
            'results_path': 'results/2-tests/a_vanilla_gan_1d/',
        },
    }


    if one_run:
        simple_main(hyperparameters=hyperparameters)
    else:
        complex_main(hyperparameters=hyperparameters)





