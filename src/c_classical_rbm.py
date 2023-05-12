import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import concurrent.futures

class ClassicalRBM:
    def __init__(self, hyperparameters):
        self.num_visible = hyperparameters['network']['num_visible']
        self.num_hidden = hyperparameters['network']['num_hidden']
        self.epochs = hyperparameters['training']['epochs']
        self.lr = hyperparameters['training']['lr']
        self.verbose = hyperparameters['training']['verbose']
        self.weights = np.random.normal(0, 0.1, (self.num_visible, self.num_hidden))
        self.visible_biases = np.zeros(self.num_visible)
        self.hidden_biases = np.zeros(self.num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_prob_hidden(self, visible_vector):
        return self.sigmoid(self.hidden_biases + np.dot(visible_vector, self.weights))

    def calc_prob_visible(self, hidden_vector):
        return self.sigmoid(self.visible_biases + np.dot(hidden_vector, self.weights.T))

    def sample_hidden(self, visible_vector):
        hidden_probs = self.calc_prob_hidden(visible_vector)
        return np.random.binomial(1, hidden_probs)

    def sample_visible(self, hidden_vector):
        visible_probs = self.calc_prob_visible(hidden_vector)
        return np.random.binomial(1, visible_probs)

    def train(self, training_data, num_reads=1):
        self.errors = []

        for epoch in range(self.epochs):
            # Shuffle the training data for each epoch
            np.random.shuffle(training_data)
            total_error = 0

            for sample in training_data:
                # Sample the hidden layer based on visible layer samples
                hidden_sample = self.sample_hidden(sample)

                # Sample the visible layer based on hidden layer samples
                visible_sample = self.sample_visible(hidden_sample)

                # Compute the probabilities for the hidden layer
                hidden_probs = self.calc_prob_hidden(sample)

                # Update the weights and biases
                hidden_2 = False
                if hidden_2:
                    hidden_sample_2 = self.sample_hidden(visible_sample)
                    self.weights += self.lr * (np.outer(sample, hidden_probs) - np.outer(visible_sample, hidden_sample_2))
                    self.visible_biases += self.lr * (sample - visible_sample)
                    self.hidden_biases += self.lr * (hidden_probs - hidden_sample_2)
                else:
                    self.weights += self.lr * (np.outer(sample, hidden_probs) - np.outer(visible_sample, hidden_sample))
                    self.visible_biases += self.lr * (sample - visible_sample)
                    self.hidden_biases += self.lr * (hidden_probs - hidden_sample)

                # Update the total error
                total_error += np.mean((sample - visible_sample) ** 2)

            total_error /= len(training_data)
            self.errors.append(total_error)

            # Print the mean squared error for the current epoch
            if self.verbose:    
                print(f"Epoch {epoch + 1}/{self.epochs}, Mean Squared Error: {total_error}")

        return sum(self.errors) / len(self.errors)

    def generate_sample(self):
        # Create a random initial visible state
        initial_state = np.random.randint(2, size=self.num_visible)

        # Sample the hidden layer based on the initial visible state
        hidden_sample = self.sample_hidden(initial_state)

        # Sample the visible layer based on the hidden layer sample
        generated_visible_sample = self.sample_visible(hidden_sample)

        return generated_visible_sample

    def generate_samples(self, n_samples):
        samples = [self.generate_sample() for _ in range(n_samples)]
        samples_array = np.array(samples)
        samples_tensor = tf.convert_to_tensor(samples_array)
        return samples_tensor



def preprocess_data(data):
    data = data / 255.0
    print(data[0][:20])
    data = (data > 0.5).astype(int)
    data = data.reshape(-1, 784)
    return data

def load_mnist(n_images, digits=None):
    (x_train, y_train), (_, _) = mnist.load_data()
    
    if digits is not None:
        digit_indices = np.isin(y_train, digits)
        x_train = x_train[digit_indices]
        y_train = y_train[digit_indices]
    
    return x_train[:n_images]

def generate_image(sample, hyperparameters):
    # Display the generated image
    new_image = sample.reshape(28, 28)
    plt.imshow(new_image, cmap='gray')
    plt.savefig(f'{hyperparameters["plotting"]["folder_path"]}epochs_{hyperparameters["training"]["epochs"]}-n_images_{n_images}-lr_{hyperparameters["training"]["lr"]}.png')
    plt.close()


def main(hyperparameters, n_images):
    training_data = load_mnist(n_images, digits=[0])#, 1])
    training_data = preprocess_data(training_data)

    rbm = ClassicalRBM(hyperparameters=hyperparameters)
    rbm.train(training_data)
    new_sample = rbm.generate_sample()

    generate_image(new_sample, hyperparameters)

if __name__ == "__main__":
    n_images = 4
    hyperparameters = {
        'network': {
            'num_visible': 784,
            'num_hidden': 20
        },
        'training': {
            'epochs': 100,
            'lr': 0.1,
            'verbose': True,
        },
        'plotting': {
            'folder_path': 'results/2-tests/c_classical_rbm/'
        }
    }
    main(hyperparameters=hyperparameters, n_images=n_images)
