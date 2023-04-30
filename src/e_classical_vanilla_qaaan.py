import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json
from scipy.stats import norm, uniform, cauchy, pareto
from a_vanilla_gan_1d import *
from c_classical_rbm import *

class ClassicalQAAAN(GAN):
    def __init__(self, hyperparameters):
        self.feature_layer_size = hyperparameters['hyperparameters_qaaan']['feature_layer_size']
        self.update_ratios = hyperparameters['hyperparameters_qaaan']['update_ratios']

        super().__init__(hyperparameters['hyperparameters_gan'])
        self.hyperparameters = hyperparameters 

        # Create a new model to extract intermediate layer output
        self.feature_layer_model = self.create_feature_layer_model(layer_index=-2)

        # Create the Restricted Boltzmann Machine which will work as Prior to the Generator
        self.rbm = ClassicalRBM(hyperparameters=hyperparameters['hyperparameters_rbm'])

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

    def train(self):
        real_data = self.sample_real_data()
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0

        for epoch in range(self.epochs):

            # Discriminator Training
            for _ in range(self.update_ratios['discriminator']):
                # Train discriminator on real data
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                rbm_prior = self.rbm.generate_samples(self.batch_size)
                generated_data = self.generator.predict(rbm_prior, verbose=0)
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # RBM Training
            for _ in range(self.update_ratios['rbm']):
                feature_layer_output = self.feature_layer_model.predict(real_data, verbose=0)
                self.rbm.train(feature_layer_output)            

            # Generator Training
            for _ in range(self.update_ratios['generator']):
                rbm_prior = self.rbm.generate_samples(self.batch_size)
                g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            g_loss_total += g_loss


            # Get intermediate layer outputs
            intermediate_output_real = self.feature_layer_model.predict(real_data, verbose=0)
            intermediate_output_fake = self.feature_layer_model.predict(generated_data, verbose=0)


            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                # print('Real Data Input to Discriminator\'s shape: ', real_data.shape)
                # print('Intermediate Output Real\'s Shape: ', intermediate_output_real.shape)
                # print('Intermediate Output Fake\'s Shape: ', intermediate_output_fake.shape)
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real_total}, D_loss_fake: {d_loss_fake_total}, G_loss: {g_loss_total}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0

    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': self.hyperparameters,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
   

def simple_main(hyperparameters):
    gan = ClassicalQAAAN(hyperparameters)
    gan.train()
    gan.plot_and_save()

def complex_main(hyperparameters):
        
    qaaan = ClassicalQAAAN(hyperparameters)
    qaaan.train()
    qaaan.plot_and_save()

    gan = GAN(hyperparameters['hyperparameters_gan'])
    gan.train()
    gan.plot_and_save()



if __name__ == "__main__":
    one_run = False

    hyperparameters_qaaan = {
        'feature_layer_size': 11,
        'update_ratios': {
            'discriminator': 5,
            'generator': 1,
            'rbm': 1,
        }
    }

    hyperparameters_gan = {
        'training': {
            'epochs': 1000,
            'batch_size': 128,
            'save_frequency': 10,
            'update_ratio_critic': hyperparameters_qaaan['update_ratios']['discriminator'],
            'learning_rate': 0.001,
        },
        'network': {
            'latent_dim': hyperparameters_qaaan['feature_layer_size'],
            'layers_gen': [2, 13, 7, 1],
            'layers_disc': [11, 29, 11, 1],
        },
        'distributions': {
            'mean': 7,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            'results_path': 'results/2-tests/e_classical_vanilla_qaaan/',
        },
    }

    hyperparameters_rbm = {
        'network': {
            'num_visible': hyperparameters_qaaan['feature_layer_size'],
            'num_hidden': 20,
        },
        'training': {
            'epochs': 1,
            'lr': 0.001,
            'lr_decay': 0.1,
            'epoch_drop': None,
            'momentum': 0,
            'batch_size': None,
            'n_images': 10,
            'verbose': False,
        },
        'plotting': {
            'folder_path': hyperparameters_gan['plotting']['results_path'],
        }
    }

    hyperparameters = {
        'hyperparameters_gan': hyperparameters_gan,
        'hyperparameters_rbm': hyperparameters_rbm,
        'hyperparameters_qaaan': hyperparameters_qaaan,
    }


    if one_run:
        simple_main(hyperparameters=hyperparameters)
    else:
        complex_main(hyperparameters=hyperparameters)





