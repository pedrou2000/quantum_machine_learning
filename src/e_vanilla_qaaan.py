import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json
from scipy.stats import norm, uniform, cauchy, pareto, beta
from a_vanilla_gan_1d import *
from c_classical_rbm import *
from d_quantum_rbm import *

class ClassicalQAAAN(GAN):
    def __init__(self, hyperparameters):
        self.feature_layer_size = hyperparameters['hyperparameters_qaaan']['network']['feature_layer_size']
        self.update_ratios = hyperparameters['hyperparameters_qaaan']['training']['update_ratios']
        self.rbm_type = hyperparameters['hyperparameters_qaaan']['network']['rbm_type']
        self.train_rbm_every_n = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_every_n']
        self.samples_train_rbm = hyperparameters['hyperparameters_qaaan']['training']['samples_train_rbm']
        self.train_rbm_cutoff_epoch = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_cutoff_epoch']
        self.train_rbm_start_epoch = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_start_epoch']

        super().__init__(hyperparameters['hyperparameters_gan'])
        self.hyperparameters = hyperparameters

        # Create a new model to extract intermediate layer output
        self.feature_layer_model = self.create_feature_layer_model(layer_index=-2)
        self.rbm_losses = []

        # Create the Restricted Boltzmann Machine which will work as Prior to the Generator
        if self.rbm_type == 'classical':
            self.rbm = ClassicalRBM(hyperparameters=hyperparameters['hyperparameters_rbm'])
        elif self.rbm_type == 'simulated' or self.rbm_type == 'quantum':
            self.rbm = QuantumRBM(hyperparameters=hyperparameters['hyperparameters_rbm'])

    def create_discriminator(self):
        model = keras.Sequential()

        model.add(layers.Dense(self.layers_disc[0], activation='relu', input_shape=(1,)))
        for n_layer in self.layers_disc[1:-2]:
            model.add(layers.Dense(n_layer, activation='relu'))
        
        # Feature Layer is the previous to the last layer of the discriminator.
        model.add(layers.Dense(self.feature_layer_size, activation='tanh'))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def create_feature_layer_model(self, layer_index):
        input_layer = self.discriminator.input
        intermediate_layer = self.discriminator.get_layer(index=layer_index).output
        return keras.Model(inputs=input_layer, outputs=intermediate_layer)
    
    def preprocess_rbm_input(self, data):
        data = (data > 0).astype(int)
        return data
    
    def reparameterize_vector(self, z, reparam_type='gaussian_noise', alpha=1, beta_alpha=0.5, beta_beta=0.5, noise_std_dev=0.3):
        zeta = np.zeros_like(z, dtype=float)
        u = np.random.uniform(0, 1, size=z.shape)

        if reparam_type == 'paper':
            idx = z == 1
            zeta[idx] = 1 - (1 / alpha) * np.log(1 - (1 - np.exp(-2 * alpha)) * u[idx])
            zeta[~idx] = -1
        elif reparam_type == 'gaussian_noise':
            mapped_samples = 2*z-1
            # print('z', z)
            noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=mapped_samples.shape)
            zeta = mapped_samples + noise
            # print('reparam z', zeta)
        else:
            raise ValueError("Invalid reparam_type. Accepted values are 'paper', 'beta', and 'gaussian_noise'")

        return zeta

        
    def generate_prior(self, n_samples=None, n_batches=None, n_lists=None):
        if n_samples is None:
            if n_batches is None:
                total_samples = self.batch_size
            else:
                total_samples = self.batch_size * n_batches
        else:
            total_samples = n_samples

        if n_lists is not None:
            total_samples *= n_lists

        # Generate samples in chunks of max_sampling and concatenate them
        max_sampling = 500
        chunks = total_samples // max_sampling
        remainder = total_samples % max_sampling

        if chunks > 0:
            samples = self.rbm.generate_samples(max_sampling)
            rbm_prior_chunks = [samples for _ in range(chunks)]
            if remainder > 0:
                rbm_prior_chunks.append(samples[:remainder])
        else:
            samples = self.rbm.generate_samples(total_samples)
            rbm_prior_chunks = [samples]

        rbm_prior = np.concatenate(rbm_prior_chunks)
        rbm_prior = self.reparameterize_vector(rbm_prior, reparam_type='gaussian_noise', alpha=1)

        # If n_batches and n_lists are specified, reshape the samples into a list of lists of batches
        if n_batches is not None and n_lists is not None:
            list_of_batches = [
                [
                    rbm_prior[k * self.batch_size:(k + 1) * self.batch_size]
                    for k in range(i * n_batches, (i + 1) * n_batches)
                ]
                for i in range(n_lists)
            ]
            return list_of_batches
        elif n_batches is not None:
            batches = [rbm_prior[i * self.batch_size:(i + 1) * self.batch_size] for i in range(n_batches)]
            return batches
        else:
            return rbm_prior

    def train(self):
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))
        num_reads = 10

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0
        rbm_loss_total = 0
        rbm_loss = 0

        rbm_prior_discriminator = self.generate_prior(n_batches=self.update_ratios['discriminator'])

        for epoch in range(self.epochs):

            # Discriminator Training
            for j in range(self.update_ratios['discriminator']):
                # Train discriminator on real data
                real_data = self.sample_real_data()
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                rbm_prior = rbm_prior_discriminator[j]
                generated_data = self.generator.predict(rbm_prior, verbose=0)
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # RBM Training
            if self.train_rbm_every_n is None or epoch % self.train_rbm_every_n == 0: 
                for _ in range(self.update_ratios['rbm']):
                    if epoch < self.train_rbm_cutoff_epoch and epoch >= self.train_rbm_start_epoch:
                        # Generate Input for RBM Training
                        random_indices = np.random.choice(len(real_data), self.samples_train_rbm, replace=False)
                        feature_layer_output = self.feature_layer_model.predict(real_data[random_indices], verbose=0)
                        rbm_input = self.preprocess_rbm_input(feature_layer_output)

                        rbm_loss = self.rbm.train(rbm_input, num_reads=10)  

                    # Generate Training Data for Generator and Discriminator
                    rbm_prior = self.generate_prior(n_batches=self.update_ratios['discriminator'] + self.update_ratios['generator'])
                    rbm_prior_discriminator = rbm_prior[:self.update_ratios['discriminator']]
                    rbm_prior_generator = rbm_prior[self.update_ratios['discriminator']:]
                    
            # Generator Training
            for j in range(self.update_ratios['generator']):
                rbm_prior = rbm_prior_generator[j]
                g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

            # Update Losses
            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            g_loss_total += g_loss
            rbm_loss_total += rbm_loss

            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                rbm_loss_total /= self.save_frequency
                print(f"Epoch {epoch}, Discriminator Loss on Real Data: {d_loss_real_total:.3f}, Discriminator Loss on Fake Data: {d_loss_fake_total:.3f}, Generator Loss: {g_loss_total:.3f}, RBM Loss: {rbm_loss_total:.3f}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)
                self.rbm_losses.append(rbm_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0
                rbm_loss_total = 0

    def train_2(self):
        real_data = self.sample_real_data()
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0

        # Train Simple GAN First
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
        
        # Throw away the trained Generator 
        # self.generator = self.create_generator()
        # self.gan = self.create_gan()
        # self.compile_models()

        # Train the RBM which will serve as Prior
        print('Training the RBM')
        feature_layer_output = self.feature_layer_model.predict(real_data[:], verbose=0)
        rbm_input = self.preprocess_rbm_input(feature_layer_output)
        rbm_loss = self.rbm.train(rbm_input) 
        print('Final RBM loss:', rbm_loss) 

        # Train the Generator Alone against Discriminator
        # print('Training Generator Alone')
        # for _ in range(self.epochs):
        #     rbm_prior = self.generate_prior()
        #     g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

        # Train Complete GAN with RBM as Prior
        for epoch in range(self.epochs):
            for _ in range(self.update_ratio_critic):
                # Train discriminator on real data
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                rbm_prior = self.generate_prior()
                generated_data = self.generator.predict(rbm_prior, verbose=0)
                fake_labels = np.zeros((self.batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # Train generator
            rbm_prior = self.generate_prior()
            g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

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

    
    def plot_losses(self, folder_path):
        plt.plot(self.rbm_losses, label='RBM Loss')
        plt.plot(self.d_losses_real, label='Discriminator Loss on Real Data')
        plt.plot(self.d_losses_fake, label='Discriminator Loss on Generated Data')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.xlabel('Epoch x '+str(self.save_frequency))
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses')
        plt.ylim(0, 2)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def wasserstein_distance(self, sample_size):
        rbm_prior = self.generate_prior(n_samples=1000)
        gen_samples = self.generator.predict(rbm_prior, verbose=0).flatten()
        real_samples = self.sample_real_data(plot=sample_size)
        return wasserstein_distance(real_samples, gen_samples)
    
    def plot_and_save(self):
        folder_path = self.create_result_folder()

        rbm_prior = self.generate_prior(n_samples=10000)[:1000]
        generated_data = self.generator.predict(rbm_prior, verbose=0).flatten()
        wasserstein_dist = self.wasserstein_distance(self.plot_size)
        self.plot_results_pdf(folder_path, generated_data, wasserstein_dist)
        self.plot_results_old(folder_path, generated_data, wasserstein_dist)

        self.plot_losses(folder_path)
        self.save_parameters_to_json(folder_path, wasserstein_dist)


def simple_main(hyperparameters):
    gan = ClassicalQAAAN(hyperparameters)
    gan.train()
    gan.plot_and_save()

def complex_main(hyperparameters):
    feature_layer_sizes = [1, 5, 10, 20]
    num_hiddens = feature_layer_sizes
    hyperparameters['hyperparameters_gan']['plotting']['results_path'] = 'results/2-tests/e_vanilla_qaaan/f_classical_different_rbm_sizes/'

    for i in feature_layer_sizes:
        hyperparameters['hyperparameters_qaaan']['feature_layer_size'] = i
        hyperparameters['hyperparameters_gan']['network']['latent_dim'] = i
        hyperparameters['hyperparameters_rbm']['network']['num_visible'] = i
        for j in num_hiddens:
            hyperparameters['hyperparameters_rbm']['network']['num_hidden'] = j 

            gan = ClassicalQAAAN(hyperparameters)
            gan.train()
            gan.plot_and_save()


def create_hyperparameters_gan(hyperparams_qaaan):
    return {
        'training': {
            'epochs': hyperparams_qaaan['training']['total_epochs'],
            'batch_size': hyperparams_qaaan['training']['batch_size'],
            'save_frequency': hyperparams_qaaan['training']['save_frequency'],
            'update_ratio_critic': hyperparams_qaaan['training']['update_ratios']['discriminator'],
            'learning_rate': hyperparams_qaaan['training']['gan_learning_rate'],
        },
        'network': {
            'latent_dim': hyperparams_qaaan['network']['feature_layer_size'],
            'layers_gen': hyperparams_qaaan['network']['layers_generator'],
            'layers_disc': hyperparams_qaaan['network']['layers_discriminator'],
        },
        'distributions': {
            'mean': hyperparams_qaaan['distributions']['mean'],
            'variance': hyperparams_qaaan['distributions']['variance'],
            'target_dist': hyperparams_qaaan['distributions']['target_dist'],
            'input_dist': hyperparams_qaaan['distributions']['input_dist'],
        },
        'plotting': {
            'plot_size': hyperparams_qaaan['plotting']['plot_size'],
            'n_bins': hyperparams_qaaan['plotting']['n_bins'],
            'results_path': hyperparams_qaaan['plotting']['results_path'],
        },
    }

def create_hyperparameters_rbm(hyperparams_qaaan):
    return {
        'network': {
            'num_visible': hyperparams_qaaan['network']['feature_layer_size'],
            'num_hidden': hyperparams_qaaan['network']['rbm_num_hidden'],
            'qpu': True if hyperparams_qaaan['network']['rbm_type'] == 'quantum' else False,
        },
        'training': {
            'epochs': hyperparams_qaaan['training']['rbm_epochs'],
            'lr': hyperparams_qaaan['training']['rbm_learning_rate'],
            'verbose': hyperparams_qaaan['training']['rbm_verbose'],
        },
        'plotting': {
            'folder_path': hyperparams_qaaan['plotting']['rbm_folder_path'],
        }
    }


if __name__ == "__main__":
    one_run = True

    hyperparameters_qaaan = {
        'training': {
            'update_ratios': {
                'discriminator': 5,
                'generator': 1,
                'rbm': 1,
            },
            'total_epochs': 100,
            'train_rbm_every_n': 1,
            'train_rbm_cutoff_epoch': 100,
            'train_rbm_start_epoch': 10,
            'samples_train_rbm': 5,
            'batch_size': 100,
            'save_frequency': 1,
            'gan_learning_rate': 0.001,
            'rbm_learning_rate': 0.001,
            'rbm_epochs': 1,
            'rbm_verbose': False,
        },
        'network': {
            'rbm_type': 'quantum',  # Can be classical, simulated, or quantum.
            'feature_layer_size': 50,
            'rbm_num_hidden': 50,
            'layers_generator': [2, 13, 7, 1],
            'layers_discriminator': [11, 29, 11, 1],
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            'rbm_folder_path': None,
        },
        'distributions': {
            'mean': 1,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
    }

    hyperparameters_qaaan['plotting']['results_path'] = 'results/2-tests/e_vanilla_qaaan/' + hyperparameters_qaaan['network']['rbm_type'] + '/'

    hyperparameters_gan = create_hyperparameters_gan(hyperparameters_qaaan)
    hyperparameters_rbm = create_hyperparameters_rbm(hyperparameters_qaaan)


    hyperparameters = {
        'hyperparameters_qaaan': hyperparameters_qaaan,
        'hyperparameters_gan': hyperparameters_gan,
        'hyperparameters_rbm': hyperparameters_rbm,
    }


    if one_run:
        simple_main(hyperparameters=hyperparameters)
    else:
        complex_main(hyperparameters=hyperparameters)


