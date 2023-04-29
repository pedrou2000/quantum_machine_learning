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
        super().__init__(hyperparameters['hyperparameters_rbm'])

        # Create a new model to extract intermediate layer output
        self.intermediate_layer_model = self.create_intermediate_layer_model(layer_index=-2)



    def create_intermediate_layer_model(self, layer_index):
        input_layer = self.discriminator.input
        intermediate_layer = self.discriminator.get_layer(index=layer_index).output
        return keras.Model(inputs=input_layer, outputs=intermediate_layer)

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


            # Get intermediate layer outputs
            intermediate_output_real = self.intermediate_layer_model.predict(real_data, verbose=0)
            intermediate_output_fake = self.intermediate_layer_model.predict(generated_data, verbose=0)


            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                print('Real Data Input to Discriminator\'s shape: ', real_data.shape)
                print('Intermediate Output Real\'s Shape: ', intermediate_output_real.shape)
                print('Intermediate Output Fake\'s Shape: ', intermediate_output_fake.shape)
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real_total}, D_loss_fake: {d_loss_fake_total}, G_loss: {g_loss_total}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0


def simple_main(hyperparameters):
    gan = ClassicalQAAAN(hyperparameters)
    gan.train()
    gan.plot_and_save()

def complex_main(hyperparameters):
    update_ratio_critics = [1,2,3,4,5]

    for update_ratio_critic in update_ratio_critics:
        hyperparameters['training']['update_ratio_critic'] = update_ratio_critic
        
        gan = ClassicalQAAAN(hyperparameters)
        gan.train()
        gan.plot_and_save()



if __name__ == "__main__":
    one_run = True

    hyperparameters_gan = {
        'training': {
            'epochs': 100,
            'batch_size': 128,
            'save_frequency': 10,
            'update_ratio_critic': 5,
        },
        'network': {
            'latent_dim': 100,
            'layers_gen': [2, 13, 7, 1],
            'layers_disc': [11, 29, 11, 1],
        },
        'distributions': {
            'mean': 1,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            'results_path': 'results/2-tests/4-qaaan/1-classical_vanilla_qaaan/',
        },
    }

    hyperparameters_rbm = {
        'training': {
            'epochs': 100,
            'batch_size': 128,
            'save_frequency': 10,
            'update_ratio_critic': 5,
        },
        'network': {
            'latent_dim': 100,
            'layers_gen': [2, 13, 7, 1],
            'layers_disc': [11, 29, 11, 1],
        },
        'distributions': {
            'mean': 1,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            'results_path': 'results/2-tests/4-qaaan/1-classical_vanilla_qaaan/',
        },
    }

    hyperparameters = {
        'hyperparameters_gan': hyperparameters_gan,
        'hyperparameters_rbm': hyperparameters_rbm,
    }


    if one_run:
        simple_main(hyperparameters=hyperparameters)
    else:
        complex_main(hyperparameters=hyperparameters)





