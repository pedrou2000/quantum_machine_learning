import os
import json
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import norm, uniform, cauchy, pareto
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from a_vanilla_gan_1d import *


class MMD_GAN(GAN):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters['hyperparameters_gan']
        self.latent_dim = self.hyperparameters['network']['latent_dim']
        self.gen_hidden_units = self.hyperparameters['network']['gen_hidden_units']
        self.critic_hidden_units = self.hyperparameters['network']['critic_hidden_units']
        self.gen_lr = self.hyperparameters['training']['lr']
        self.critic_lr = self.hyperparameters['training']['lr']
        self.epochs = self.hyperparameters['training']['epochs']
        self.batch_size = self.hyperparameters['training']['batch_size']
        self.update_ratio_critic = self.hyperparameters['training']['update_ratio_critic']
        self.update_ratio_gen = self.hyperparameters['training']['update_ratio_gen']
        self.mean = self.hyperparameters['distributions']['mean']
        self.variance = self.hyperparameters['distributions']['variance']
        self.results_path = self.hyperparameters['plotting']['results_path']
        self.save_frequency = self.hyperparameters['training']['save_frequency']
        self.mmd_lamb = self.hyperparameters['training']['mmd_lamb']
        self.sigmas = self.hyperparameters['training']['sigmas']
        self.clip = self.hyperparameters['training']['clip']
        self.target_dist = self.hyperparameters['distributions']['target_dist']
        self.input_dist = self.hyperparameters['distributions']['input_dist']
        self.plot_size = self.hyperparameters['plotting']['plot_size']
        self.n_bins = self.hyperparameters['plotting']['n_bins']

        self.generator_losses = []
        self.critic_losses = []
        self.wasserstein_dists = []

        self.generator = self.create_generator()
        self.critic = self.create_critic()

        self.generator_optimizer = Adam(learning_rate=self.gen_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
    
    def sample_noise(self, plot=False):
        size = self.batch_size
        if plot:
            size = plot
        if self.input_dist == "gaussian":
            return np.random.normal(0, 1, (size, self.latent_dim)).astype(np.float32)
        elif self.input_dist == "uniform":
                return np.random.uniform(0, 1,(size, self.latent_dim)).astype(np.float32)

    def create_generator(self):
        model = Sequential()
        model.add(Dense(self.gen_hidden_units[0], input_dim=self.latent_dim))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(1))
        return model

    def create_critic(self):
        model = Sequential()
        model.add(Dense(self.critic_hidden_units[0], input_dim=1))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(1))
        return model


    def gaussian_kernel_matrix(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        x = tf.reshape(x, (x_size, 1, dim))
        y = tf.reshape(y, (1, y_size, dim))
        tile = tf.tile(x, (1, y_size, 1))
        diff = tile - y

        sum_squares = tf.reduce_sum(tf.square(diff), axis=-1)
        sum_squares = tf.expand_dims(sum_squares, axis=-1)

        beta = 1.0 / (2.0 * tf.cast(self.sigmas, tf.float32))
        exponent = -beta * sum_squares

        return tf.reduce_sum(tf.exp(exponent), axis=-1)

    def compute_mmd(self, x_real, x_fake):
        xx = self.gaussian_kernel_matrix(x_real, x_real)
        yy = self.gaussian_kernel_matrix(x_fake, x_fake)
        xy = self.gaussian_kernel_matrix(x_real, x_fake)
        n = tf.cast(tf.shape(x_real)[0], dtype=tf.float32)
        mmd = (tf.reduce_sum(xx) - tf.linalg.trace(xx)) / (n * (n - 1))
        mmd -= 2 * tf.reduce_sum(xy) / (n * n)
        mmd += (tf.reduce_sum(yy) - tf.linalg.trace(yy)) / (n * (n - 1))
        return mmd

    def train(self):
        average_critic_loss = 0
        average_gen_loss = 0
        wasserstein_dist = 0

        for epoch in range(self.epochs):

            # Critic Training Step
            for _ in range(self.update_ratio_critic):
                x_real = self.sample_real_data()
                z = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake = self.generator(z)
                    critic_x_real = self.critic(tf.expand_dims(x_real, -1))
                    critic_x_fake = self.critic(x_fake)
                    mmd = self.compute_mmd(critic_x_real, critic_x_fake)
                    constraint_term = tf.reduce_mean(critic_x_real) - tf.reduce_mean(critic_x_fake)
                    critic_loss = -mmd + self.mmd_lamb * tf.minimum(constraint_term, 0)

                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                average_critic_loss += critic_loss

                # Clip critic weights
                for var in self.critic.trainable_variables:
                    var.assign(tf.clip_by_value(var, -self.clip, self.clip))

            # Generator Training Step
            for _ in range(self.update_ratio_gen):
                z = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake = self.generator(z)
                    critic_x_fake = self.critic(x_fake)
                    mmd = self.compute_mmd(critic_x_real, critic_x_fake)
                    generator_loss = mmd
                generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                average_gen_loss += generator_loss
            
            wasserstein_dist += self.wasserstein_distance(1000)#self.plot_size)

            # Registering Results
            if epoch % self.save_frequency == 0:
                average_gen_loss /= self.save_frequency
                average_critic_loss /= self.save_frequency
                wasserstein_dist /= self.save_frequency
                self.generator_losses.append(average_gen_loss)
                self.critic_losses.append(average_critic_loss)
                self.wasserstein_dists.append(wasserstein_dist)
                print(f"Epoch {epoch}: generator MMD = {average_gen_loss:.4f}, critic MMD = {average_critic_loss:.4f}")
                average_gen_loss = 0
                average_critic_loss = 0
                wasserstein_dist = 0


    def plot_losses(self, folder_path):
        plt.plot(self.critic_losses, label='Critic MMD')
        plt.plot(self.generator_losses, label='Generator MMD')
        # plt.plot(self.wasserstein_dists, label='Wasserstein Distance')
        plt.xlabel('Epoch / '+str(self.save_frequency))
        plt.ylabel('MMD Loss')
        plt.legend()
        # plt.title('MMDs')

        margin_axis = 0.09
        margin_no_axes = -0.01
        plt.subplots_adjust(left=margin_axis, right=1+margin_no_axes, bottom=margin_axis, top=1+margin_no_axes-0.01)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
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
        generated_data = self.generator(noise).numpy().flatten()
        self.wasserstein_dist = self.wasserstein_distance(self.plot_size, gen_samples=generated_data)
        self.plot_results_pdf(folder_path, generated_data, self.wasserstein_dist)
        self.plot_results_old(folder_path, generated_data, self.wasserstein_dist)

        self.plot_losses(folder_path)
        self.plot_wasserstein(folder_path)
        self.save_parameters_to_json(folder_path, self.wasserstein_dist)



if __name__ == "__main__":
    GANClass = MMD_GAN
    main_type = 'distributions' # can be one_run, update_ratios, lr, distributions
    num_runs = 3

    hyperparameters = {
        'hyperparameters_gan': {

            'training': {
                'epochs': 10000,
                'save_frequency': 100,
                'batch_size': 64,
                'update_ratio_critic': 2,
                'update_ratio_gen': 1,
                'lr': 0.0001,
                'mmd_lamb': 0.01,
                'clip': 1,
                'sigmas': [1, 2, 4, 8, 16]
            },
            'network': {
                'latent_dim': 2,
                'gen_hidden_units': [7, 13, 7],
                'critic_hidden_units': [11, 29, 11]
            },
            'distributions': {
                'mean': 1,
                'variance': 3,
                'target_dist': 'pareto', # can be uniform, gaussian, pareto or cauchy
                'input_dist': 'uniform'
            },
            'plotting': {
                'plot_size': 10000,
                'n_bins': 100,
                'results_path': 'results/5-extra_tests/b_mmd_gan_1d/1-mean_21/different_'+ main_type + '/',
                # 'results_path': 'results/5-extra_tests/b_mmd_gan_1d/0-tests/',
            }
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
