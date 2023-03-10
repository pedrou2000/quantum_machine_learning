import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import imageio.v2 as imageio
import glob
import time
import tensorflow_docs.vis.embed as embed
from scipy.stats import wasserstein_distance
#print(wasserstein_distance([0, 1, 3], [5, 6, 8]))

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from utils import plot_distributions, create_gif


class gan():
    def __init__(self, training_sets, dimension, num_samples, input_shape_gen, output_shape_gen, num_layers, 
                 num_neurons_per_layer, use_bias):
        # Optimizers: The discriminator and the generator optimizers are different since you will train two networks separately.
        self.training_sets = training_sets
        self.dimension = dimension 
        self.num_samples = num_samples 

        self.input_shape_gen = input_shape_gen 
        self.output_shape_gen = output_shape_gen 
        self.num_layers = num_layers 
        self.num_neurons_per_layer = num_neurons_per_layer 
        self.use_bias = use_bias

        self._make_generator_model() 
        self._make_discriminator_model() 

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def _make_generator_model(self):
        self.generator = tf.keras.Sequential()

        self.generator.add(layers.Dense(self.num_neurons_per_layer, use_bias=self.use_bias, input_dim=input_shape_gen))
        #self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())

        for _ in range(self.num_layers):
            self.generator.add(layers.Dense(self.num_neurons_per_layer, use_bias=self.use_bias))
            #self.generator.add(layers.BatchNormalization())
            self.generator.add(layers.LeakyReLU())

        self.generator.add(layers.Dense(self.output_shape_gen))
        assert self.generator.output_shape == (None, self.output_shape_gen)

    def _make_discriminator_model(self):
        self.discriminator = tf.keras.Sequential()

        self.discriminator.add(layers.Dense(self.num_neurons_per_layer, use_bias=self.use_bias, input_dim=self.output_shape_gen))
        #self.discriminator.add(layers.BatchNormalization())
        self.discriminator.add(layers.LeakyReLU())

        for _ in range(self.num_layers):
                self.discriminator.add(layers.Dense(self.num_neurons_per_layer, use_bias=self.use_bias))
                #self.discriminator.add(layers.BatchNormalization())
                self.discriminator.add(layers.LeakyReLU()) 

        self.discriminator.add(layers.Dense(1, use_bias=self.use_bias))
        assert self.discriminator.output_shape == (None, 1)
        #print(model.summary())


    def _discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss: This method quantifies how well the discriminator is able to distinguish real distributions from fakes. 
        It compares the discriminator's predictions on real distributions to 1, and the discriminator's predictions on 
        fake (generated) distributions to 0. """

        real_loss = (real_output-1)**2
        fake_loss = (fake_output)**2
        #print('DISCRIMINATOR LOSS (fake output: ' + str(fake_output.numpy()) + ', real output: ' + str(real_output.numpy()) + '): real loss: ' + str(real_loss.numpy()) + ', fake loss: '+str(fake_loss.numpy()))
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        """ Generator loss: The generator's loss quantifies how well it was able to trick the discriminator. 
        Intuitively, if the generator is performing well, the discriminator will classify the fake distributions 
        as real (or 1). Here, compare the discriminators decisions on the generated distributions to 1."""

        fake_loss = (fake_output-1)**2
        #print('GENERATOR LOSS (fake output: ' + str(fake_output.numpy()) + '): fake loss: '+str(fake_loss.numpy()))
        return fake_loss


    def _show_results(self, model, epoch, test_input, dataset):
        predictions = model(test_input, training=False)

        print('Epoch ' + str(epoch) + ' mean: ' + str(tf.reduce_mean(predictions).numpy()))
        plot_distributions(
            dist_1=[predictions[0]], 
            dist_2=dataset.numpy()[0][0], 
            color_dist_1 = color_generated_distribution, 
            color_dist_2 = color_real_distribution,
            images_path = images_path,
            epoch=epoch, 
            n_bins = 10,
        )


    def _train_step_generator(self, steps=10):
        noise = tf.random.uniform([dimension, self.input_shape_gen])
        #print('Noise: '+str(noise[0][0:4].numpy()))

        for step in range(steps):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_distributions = self.generator(noise, training=True)

                fake_output = self.discriminator(generated_distributions, training=True)[0][0]

                #print("Real Data Discriminator Value: " + str(real_output.numpy()[0][0]))
                #print("Generated Data Discriminator Value: " + str(fake_output.numpy()[0][0]))

                gen_loss = self._generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
    def _train_step_discriminator(self, distributions, steps=10):
        noise = tf.random.uniform([dimension, self.input_shape_gen])
        #print('Noise: '+str(noise[0][0:4].numpy()))
        for step in range(steps):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_distributions = self.generator(noise, training=True)

                real_output = self.discriminator(distributions, training=True)[0][0]
                fake_output = self.discriminator(generated_distributions, training=True)[0][0]

                #print("Real Data Discriminator Value: " + str(real_output.numpy()[0][0]))
                #print("Generated Data Discriminator Value: " + str(fake_output.numpy()[0][0]))

                disc_loss = self._discriminator_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
    def _train_step(self, distributions):
        noise = tf.random.uniform([dimension, self.input_shape_gen])
        #print('Noise: '+str(noise[0][0:4].numpy()))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_distributions = self.generator(noise, training=True)

            real_output = self.discriminator(distributions, training=True)[0][0]
            fake_output = self.discriminator(generated_distributions, training=True)[0][0]

            #print("Real Data Discriminator Value: " + str(real_output.numpy()[0][0]))
            #print("Generated Data Discriminator Value: " + str(fake_output.numpy()[0][0]))

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, seed, epochs, show_every_n_epochs):
        """ The training loop begins with generator receiving a random seed as input. That seed is used to 
        produce a distribution. The discriminator is then used to classify real distributions (drawn from the training 
        set) and fake distributions (produced by the generator). The loss is calculated for each of these models, 
        and the gradients are used to update the generator and discriminator. """
        print()
        print('TRAINING:')

        for epoch in range(epochs):
            if epoch % 100 == 0:
                print(epoch)
            start = time.time()

            for distribution in dataset:
                self._train_step(distribution)
            
            if (epoch+1) % show_every_n_epochs == 0:
                self._show_results(self.generator, epoch + 1, seed, dataset)
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        create_gif()

    def train_batched(self, dataset, seed, epochs, show_every_n_epochs):
        """ The training loop begins with generator receiving a random seed as input. That seed is used to 
        produce a distribution. The discriminator is then used to classify real distributions (drawn from the training 
        set) and fake distributions (produced by the generator). The loss is calculated for each of these models, 
        and the gradients are used to update the generator and discriminator. """
        print()
        print('TRAINING::')

        for epoch in range(epochs):
            start = time.time()
            if epoch % 100 == 0:
                print(epoch)

            for distribution in dataset:
                self._train_step_discriminator(distribution)
                self._train_step_generator()
            
            if (epoch+1) % show_every_n_epochs == 0:
                self._show_results(self.generator, epoch + 1, seed, dataset)
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        create_gif()






n_bins = 10

color_generated_distribution = "red"
color_real_distribution = "black"

results_path = '2-gan/results/'
images_path = results_path + 'images/'
gif_file = results_path + 'gan.gif'



# Initialize GAN Hyperparameters and Data

training_sets = 10
dimension = 10
num_samples = 100

input_shape_gen = num_samples
output_shape_gen = num_samples
num_layers = 4
num_neurons_per_layer = 10
use_bias = True

mean = 5
stddev = 0.2
samples_train = tf.random.normal([training_sets, dimension, num_samples], mean=mean, stddev=stddev,)
#samples_train =  tf.random.uniform([training_sets,dimension, num_samples], minval=-1, maxval=1)

seed = tf.random.uniform([dimension, input_shape_gen], minval=0, maxval=1)

epochs = 50
show_n_pictures = 50
show_every_n_epochs = epochs//show_n_pictures



# Initialize and Train GAN

my_gan = gan(
    training_sets = training_sets,
    dimension = dimension,
    num_samples = num_samples,
    input_shape_gen = input_shape_gen,
    output_shape_gen = output_shape_gen,
    num_layers = num_layers,
    num_neurons_per_layer = num_neurons_per_layer,
    use_bias=use_bias,
)

my_gan.train(samples_train, seed, epochs, show_every_n_epochs)
#my_gan.train_batched(samples_train, seed, epochs, show_every_n_epochs)
