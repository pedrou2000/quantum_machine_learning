# Source: https://github.com/GabrielBianconi/pytorch-rbm/blob/master/rbm.py

import torch
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_distributions(blue_dist, red_dist, bins = 10):
    plt.hist(x=blue_dist, bins=bins, color='blue', alpha=0.7, rwidth=0.85)
    plt.hist(x=red_dist, bins=bins, color='red', alpha=0.7, rwidth=0.85)
    plt.show()


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

    def train(self, epochs):
        print('Training RBM...')
        for epoch in range(epochs):
            epoch_error = 0.0

            for batch in input_data:
                batch_error = rbm.contrastive_divergence(batch)

                epoch_error += batch_error

            print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
    
    def sample(self, iterations, noise):
        generated_distribution = rbm.sample_hidden(noise)
        generated_distribution = rbm.sample_visible(generated_distribution)

        for _ in range(iterations):
            generated_distribution = rbm.sample_hidden(generated_distribution)
            generated_distribution = rbm.sample_visible(generated_distribution)
        return generated_distribution



# Hyperparameters and Data
num_visible_layers = 100
num_hidden_layers = 100
k = 50
epochs = 50
batches = 100

input_data = torch.randn(batches, num_hidden_layers, num_visible_layers)
noise = torch.rand(num_visible_layers)

iterations = 0


# Create, Train and Sample from RBM
rbm = RBM(num_visible_layers, num_hidden_layers, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=False)
rbm.train(epochs)
generated_distribution = rbm.sample(iterations, noise)


# Show the results
print(generated_distribution.numpy())
print(input_data[0][0].numpy())
plot_distributions(blue_dist=generated_distribution.numpy(), red_dist=input_data[0][0].numpy(), bins = 10)
