# Source: https://github.com/GabrielBianconi/pytorch-rbm/blob/master/rbm.py

import torch
import tensorflow as tf
import matplotlib.pyplot as plt

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from utils import plot_distributions, create_gif


class RBM():
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def set_plot_config(self, n_bins, color_generated_distribution, color_real_distribution, images_path, errors_path, gif_path,):
        self.n_bins = n_bins 
        self.color_generated_distribution = color_generated_distribution
        self.color_real_distribution = color_real_distribution
        self.images_path = images_path
        self.errors_path = errors_path
        self.gif_path = gif_path
    
    def _show_results(self, iterations, noise, epoch, input_data):
        generated_distribution = self.sample(iterations, noise)

        plot_distributions(
            dist_1=generated_distribution.numpy(), 
            dist_2=input_data[0][0].numpy(), 
            color_dist_1 = self.color_generated_distribution, 
            color_dist_2 = self.color_real_distribution,
            images_path = self.images_path,
            epoch=epoch, 
            n_bins=self.n_bins,
        )

    def train(self, epochs, input_data, show_every_n_epochs, noise, iterations):
        print('Training RBM...')
        for epoch in range(epochs):
            epoch_error = 0.0

            for batch in input_data:
                batch_error = rbm.contrastive_divergence(batch)

                epoch_error += batch_error
            
            if epoch % show_every_n_epochs == 0:
                print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
                self._show_results(iterations, noise, epoch, input_data)
        create_gif(images_path=self.images_path, gif_file=self.gif_path)

    
    def sample(self, iterations, noise):
        generated_distribution = rbm.sample_hidden(noise)
        generated_distribution = rbm.sample_visible(generated_distribution)

        for _ in range(iterations):
            generated_distribution = rbm.sample_hidden(generated_distribution)
            generated_distribution = rbm.sample_visible(generated_distribution)
        return generated_distribution



# Hyperparameters and Data
batches = 10
num_hidden_layers = 30
num_visible_layers = 30

k = 2
iterations = k
epochs = 1000
show_every_n_epochs = 30

input_data = torch.randn(batches, num_hidden_layers, num_visible_layers) *0.1+5
#input_data = torch.randn(batches, num_hidden_layers, num_visible_layers)
noise = torch.rand(num_visible_layers)


# Results Plot Configuration
n_bins = 10
show_n_pictures = 10
show_every_n_epochs = epochs//show_n_pictures

color_generated_distribution = "red"
color_real_distribution = "black"

results_path = 'rbm/results/1-initial_tests/'
images_path = results_path + 'images/'
errors_path = results_path + 'errors/'
gif_path = results_path + 'rbm.gif'


# Create, Train and Sample from RBM
rbm = RBM(num_visible_layers, num_hidden_layers, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=False)
rbm.set_plot_config(n_bins, color_generated_distribution, color_real_distribution, images_path, errors_path, gif_path,)
rbm.train(epochs, input_data, show_every_n_epochs, noise, iterations)