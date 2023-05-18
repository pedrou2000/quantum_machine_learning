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
from scipy.stats import wasserstein_distance
import os
import multiprocessing as mp
from functools import partial

class GAN:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters['hyperparameters_gan']
        self.latent_dim = self.hyperparameters['network']['latent_dim']
        self.epochs = self.hyperparameters['training']['epochs']
        self.batch_size = self.hyperparameters['training']['batch_size']
        self.save_frequency = self.hyperparameters['training']['save_frequency']
        self.update_ratio_critic = self.hyperparameters['training']['update_ratio_critic']
        self.learning_rate = self.hyperparameters['training']['learning_rate']
        self.mean = self.hyperparameters['distributions']['mean']
        self.variance = self.hyperparameters['distributions']['variance']
        self.target_dist = self.hyperparameters['distributions']['target_dist']
        self.input_dist = self.hyperparameters['distributions']['input_dist']
        self.plot_size = self.hyperparameters['plotting']['plot_size']
        self.n_bins = self.hyperparameters['plotting']['n_bins']
        self.results_path = self.hyperparameters['plotting']['results_path']
        self.layers_gen = self.hyperparameters['network']['layers_gen']
        self.layers_disc = self.hyperparameters['network']['layers_disc']
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.create_gan()
        self.compile_models()
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []
        self.wasserstein_dists = []

    def sample_real_data(self, plot=False):
        size = self.batch_size
        if plot:
            size = plot
        if self.target_dist == "gaussian":
            samples = np.random.normal(self.mean, self.variance, size)
        elif self.target_dist == "uniform":
            samples = np.random.uniform(self.mean, self.variance, size)
        elif self.target_dist == "cauchy":
            samples = cauchy.rvs(self.mean, self.variance, size)
        elif self.target_dist == "pareto":
            samples = pareto.rvs(b=np.sqrt(self.variance), scale=self.mean, size=size)
        return samples

    def sample_noise(self, plot=False):
        size = self.batch_size
        if plot:
            size = plot
        if self.input_dist == "gaussian":
            return np.random.normal(0, 1, (size, self.latent_dim))
        elif self.input_dist == "uniform":
            return np.random.uniform(0, 1,(size, self.latent_dim))

    def create_generator(self):
        model = keras.Sequential()
        
        model.add(layers.Dense(self.layers_gen[0], activation='elu', input_dim=self.latent_dim))
        for n_layer in self.layers_gen[1:-1]:
            model.add(layers.Dense(n_layer, activation='elu'))
        model.add(layers.Dense(self.layers_gen[-1], activation='linear'))
        return model

    def create_discriminator(self):
        model = keras.Sequential()

        model.add(layers.Dense(self.layers_disc[0], activation='elu', input_shape=(1,)))
        for n_layer in self.layers_disc[1:-1]:
            model.add(layers.Dense(n_layer, activation='elu'))
        model.add(layers.Dense(self.layers_disc[len(self.layers_disc)-1], activation='sigmoid'))
        return model

    def create_gan(self):
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def compile_models(self):
        custom_adam = Adam()#learning_rate=self.learning_rate)
        self.discriminator.compile(optimizer=custom_adam, loss='binary_crossentropy')
        self.discriminator.trainable = False
        self.gan.compile(optimizer=custom_adam, loss='binary_crossentropy')

    def train(self):
        real_labels = np.ones((self.batch_size, 1))
        real_data = self.sample_real_data()

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0
        wasserstein_dist = 0

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
            wasserstein_dist += self.wasserstein_distance(1000)#self.plot_size)

            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                wasserstein_dist /= self.save_frequency
                print(f"Epoch {epoch}, Discriminator Loss on Real Data: {d_loss_real_total:.3f}, Discriminator Loss on Fake Data: {d_loss_fake_total:.3f}, Generator Loss: {g_loss_total:.3f}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)
                self.wasserstein_dists.append(wasserstein_dist)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0
                wasserstein_dist = 0


    def plot_results_pdf(self, folder_path, generated_data, wasserstein_dist):
        plt.hist(generated_data, bins=self.n_bins, alpha=0.6, label="Generated Data", density=True) # set density=True to display percentages

        # Define bounds and pdf_values for each distribution type
        if self.target_dist == "gaussian":
            lower_bound = self.mean - 4 * np.sqrt(self.variance)
            upper_bound = self.mean + 4 * np.sqrt(self.variance)
            x_values = np.linspace(lower_bound, upper_bound, 1000)
            pdf_values = norm.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "uniform":
            lower_bound = self.mean
            upper_bound = self.variance
            x_values = np.linspace(lower_bound - 1, upper_bound + 1, 1000)
            scale = upper_bound - lower_bound
            pdf_values = uniform.pdf(x_values, loc=lower_bound, scale=scale)
        elif self.target_dist == "cauchy":
            lower_bound = self.mean - 6 * np.sqrt(self.variance)
            upper_bound = self.mean + 6 * np.sqrt(self.variance)
            x_values = np.linspace(lower_bound, upper_bound, 1000)
            pdf_values = cauchy.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "pareto":
            lower_bound = self.mean
            upper_bound = self.mean + 3 * np.sqrt(self.variance)  # Adjust this multiplier as needed
            x_values = np.linspace(lower_bound - 1, upper_bound + 1, 1000)
            pdf_values = pareto.pdf(x_values, b=np.sqrt(self.variance), scale=self.mean)


        pdf_values = pdf_values / (pdf_values.sum() * np.diff(x_values)[0]) # normalize the PDF

        plt.plot(x_values, pdf_values, label="Real PDF")

        plt.plot([], [], ' ', label=f'Wasserstein Distance: {wasserstein_dist:.2f}')
        # plt.legend()

        # Set x and y axes range
        plt.xlim(x_values[0], x_values[-1])
        plt.ylim(0, pdf_values.max() * 1.3) 
        ax = plt.gca()
        ax.set_xlabel('x')
        ax.set_ylabel('pdf')
                
        margin_axis = 0.05
        margin_no_axes = -0.01
        plt.subplots_adjust(left=margin_axis+0.04, right=1+margin_no_axes, bottom=margin_axis+0.04, top=1+margin_no_axes)

        plt.savefig(f"{folder_path}histogram_pdf.png", dpi=300)
        plt.close()

    def plot_results_old(self, folder_path, generated_data, wasserstein_dist):
        plt.hist(generated_data, bins=self.n_bins, alpha=0.6, label='Generated Data', density=True)
        # plt.ylim(0, 1)

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
            pdf_values = pareto.pdf(x_values, b=np.sqrt(self.variance), scale=self.mean)

        pdf_values = pdf_values / (pdf_values.sum() * np.diff(x_values)[0])  # normalize the PDF

        plt.plot(x_values, pdf_values, label="Real PDF")

        plt.plot([], [], ' ', label=f'Wasserstein Distance: {wasserstein_dist:.2f}')
        # plt.legend()
 
        ax = plt.gca()
        ax.set_xlabel('x')
        ax.set_ylabel('pdf')
                
        margin_axis = 0.05
        margin_no_axes = -0.01
        plt.subplots_adjust(left=margin_axis+0.04, right=1+margin_no_axes, bottom=margin_axis+0.04, top=1+margin_no_axes)

        plt.savefig(f'{folder_path}histogram_old.png', dpi=300)
        plt.close()

    def plot_losses(self, folder_path):
        plt.plot(self.d_losses_real, label='Discriminator Loss on Real Data')
        plt.plot(self.d_losses_fake, label='Discriminator Loss on Generated Data')
        plt.plot(self.g_losses, label='Generator Loss')
        # plt.plot(self.wasserstein_dists, label='Wasserstein Distance')
        plt.xlabel('Epoch / '+str(self.save_frequency))
        plt.ylabel('Loss')
        plt.legend()
        # plt.title('Losses')
        # plt.ylim(0, 2)

        margin_axis = 0.09
        margin_no_axes = -0.01
        plt.subplots_adjust(left=margin_axis, right=1+margin_no_axes, bottom=margin_axis, top=1+margin_no_axes-0.01)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def plot_wasserstein(self, folder_path):
        plt.plot(self.wasserstein_dists, label='Wasserstein Distance')
        plt.xlabel('Epoch / '+str(self.save_frequency))
        plt.ylabel('Wasserstein Distance')

        margin_axis = 0.09
        margin_no_axes = -0.01
        plt.subplots_adjust(left=margin_axis+0.04, right=1+margin_no_axes, bottom=margin_axis, top=1+margin_no_axes-0.01)

        plt.savefig(f'{folder_path}wasserstein.png', dpi=300)
        plt.close()

    def wasserstein_distance(self, sample_size, gen_samples=None):
        if gen_samples is None:
            noise = self.sample_noise(plot=sample_size)
            gen_samples = self.generator.predict(noise, verbose=0).flatten()
        real_samples = self.sample_real_data(plot=sample_size)
        return wasserstein_distance(real_samples, gen_samples)

    def save_parameters_to_json(self, folder_path, wasserstein_dist):
        parameters = {
            'wasserstein_distance': round(wasserstein_dist, 3),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': self.hyperparameters,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
    
    def create_result_folder(self):
        timestamp = time.strftime("%m%d_%H%M%S")
        folder_path = self.results_path + f'{self.target_dist}_{self.epochs}_{self.mean}_{self.variance}_{timestamp}/'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def plot_and_save(self):
        folder_path = self.create_result_folder()

        noise = self.sample_noise(plot=self.plot_size)
        generated_data = self.generator.predict(noise, verbose=0).flatten()
        self.wasserstein_dist = self.wasserstein_distance(self.plot_size, gen_samples=generated_data)
        self.plot_results_pdf(folder_path, generated_data, self.wasserstein_dist)
        self.plot_results_old(folder_path, generated_data, self.wasserstein_dist)
        
        self.plot_losses(folder_path)
        self.plot_wasserstein(folder_path)
        self.save_parameters_to_json(folder_path, self.wasserstein_dist)


def simple_main(hyperparameters, GANClass=GAN):
    gan = GANClass(hyperparameters)
    gan.train()
    gan.plot_and_save()

def main_different_update_ratios(hyperparameters, GANClass=GAN, num_runs=1):
    print('main_different_update_ratios')
    update_ratio_critics = [1,2,3,4,5]
    avg_wasserstein_distances = []
    max_wasserstein_distances = []

    results_path = hyperparameters['hyperparameters_gan']['plotting']['results_path']

    for update_ratio_critic in update_ratio_critics:
        # Update the hyperparameters
        hyperparameters['hyperparameters_gan']['training']['update_ratio_critic'] = update_ratio_critic
        hyperparameters['hyperparameters_gan']['plotting']['results_path'] = results_path + str(update_ratio_critic) + '/'

        wasserstein_distances = []
        for _ in range(num_runs):
            # Train the model
            gan = GANClass(hyperparameters)
            gan.train()
            gan.plot_and_save()
            wasserstein_distances.append(gan.wasserstein_dist)

        avg_wasserstein_distances.append(sum(wasserstein_distances) / num_runs)
        max_wasserstein_distances.append(max(wasserstein_distances))

    # Plotting the results
    plt.figure(figsize=(10,6))
    plt.plot(update_ratio_critics, avg_wasserstein_distances, marker='o', label='Average')
    plt.plot(update_ratio_critics, max_wasserstein_distances, marker='o', label='Maximum')
    plt.xlabel('Update Ratio Critic')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance vs Update Ratio Critic')
    plt.grid(True)
    plt.legend()

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.95)

    # Save the plot
    plt.savefig(f'{results_path}wasserstein_distance_vs_update_ratio.png', dpi=300)
    plt.close()

    print("Plot saved successfully")

def main_different_lrs(hyperparameters, GANClass=GAN, num_runs=1):
    print('main_different_lrs')
    lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    avg_wasserstein_distances = []
    max_wasserstein_distances = []

    results_path = hyperparameters['hyperparameters_gan']['plotting']['results_path']

    for learning_rate in lrs:
        # Update the hyperparameters
        hyperparameters['hyperparameters_gan']['training']['learning_rate'] = learning_rate
        hyperparameters['hyperparameters_gan']['plotting']['results_path'] = results_path + str(learning_rate) + '/'

        wasserstein_distances = []
        for _ in range(num_runs):
            # Train the model
            gan = GANClass(hyperparameters)
            gan.train()
            gan.plot_and_save()
            wasserstein_distances.append(gan.wasserstein_dist)

        avg_wasserstein_distances.append(sum(wasserstein_distances) / num_runs)
        max_wasserstein_distances.append(max(wasserstein_distances))

    # Plotting the results
    plt.figure(figsize=(10,6))
    plt.semilogx(lrs, avg_wasserstein_distances, marker='o', label='Average')  # Changed from plt.plot to plt.semilogx
    plt.semilogx(lrs, max_wasserstein_distances, marker='o', label='Maximum')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance vs Learning Rate')
    plt.grid(True)
    plt.legend()

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.95)
    # Save the plot
    plt.savefig(f'{results_path}wasserstein_distance_vs_lr.png', dpi=300)
    plt.close()

    print("Plot saved successfully")

def main_different_distributions(hyperparameters, GANClass=GAN, num_runs=1):
    print('main_different_distributions')
    distributions = {
        # 'gaussian': [21,1],
        'uniform': [21,23],
        # 'pareto': [1,3],
        # 'cauchy': [3,1],
    }

    results_path = hyperparameters['hyperparameters_gan']['plotting']['results_path']

    for key, value in distributions.items():
        hyperparameters['hyperparameters_gan']['plotting']['results_path'] = results_path + key + '/'
        for _ in range(num_runs):
            hyperparameters['hyperparameters_gan']['distributions']['target_dist'] = key
            hyperparameters['hyperparameters_gan']['distributions']['mean'] = value[0]
            hyperparameters['hyperparameters_gan']['distributions']['variance'] = value[1]
            print()
            print(key)
            gan = GANClass(hyperparameters)
            gan.train()
            gan.plot_and_save()
    



if __name__ == "__main__":
    GANClass = GAN
    main_type = 'distributions' # can be one_run, update_ratios, lr, distributions
    num_runs = 1

    hyperparameters = {
        'hyperparameters_gan': {
            'training': {
                'epochs': 10000,
                'batch_size': 128,
                'save_frequency': 100,
                'update_ratio_critic': 5,
                'learning_rate': 0.001,
            },
            'network': {
                'latent_dim': 2,
                'layers_gen': [7, 13, 7, 1],
                'layers_disc': [11, 29, 11, 1],
            },
            'distributions': {
                'mean': 21,
                'variance': 1,
                'target_dist': 'cauchy', # can be gaussian, uniform, pareto or cauchy
                'input_dist': 'uniform',
            },
            'plotting': {
                'plot_size': 10000,
                'n_bins': 100,
                'results_path': 'results/3-final_tests/a_vanilla_gan_1d/1-mean_21/different_'+ main_type + '/'#2-5_runs/',
                # 'results_path': 'results/5-extra_tests/a_vanilla_gan_1d/0-tests/',
            },
        }
    }


    if main_type == 'one_run':
        simple_main(hyperparameters=hyperparameters, GANClass=GANClass)
    elif main_type == 'update_ratios':
        main_different_update_ratios(hyperparameters=hyperparameters, GANClass=GANClass, num_runs=num_runs)
    elif main_type == 'lr':
        main_different_lrs(hyperparameters=hyperparameters, GANClass=GANClass, num_runs=num_runs)
    elif main_type == 'distributions':
        main_different_distributions(hyperparameters=hyperparameters, GANClass=GANClass, num_runs=num_runs)





