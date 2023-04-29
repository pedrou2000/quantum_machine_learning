import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class ClassicalRBM:
    def __init__(self, num_visible, num_hidden, epochs=50, lr=0.1, lr_decay=0.1, epoch_drop=None, momentum=0, batch_size=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epoch_drop = epoch_drop
        self.momentum = momentum
        self.batch_size = batch_size
        self.weights = np.random.normal(0, 0.1, (num_visible, num_hidden))
        self.visible_biases = np.zeros(num_visible)
        self.hidden_biases = np.zeros(num_hidden)

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

    def train(self, training_data, len_x=1, len_y=1):
        for epoch in range(self.epochs):
            # Shuffle the training data for each epoch
            np.random.shuffle(training_data)
            total_error = 0

            for sample in training_data:
                # Sample the hidden layer based on visible layer sample
                hidden_sample = self.sample_hidden(sample)

                # Sample the visible layer based on hidden layer sample
                visible_sample = self.sample_visible(hidden_sample)

                # Compute the probabilities for the hidden layer
                hidden_probs = self.calc_prob_hidden(sample)

                # Update the weights and biases
                self.weights += self.lr * (np.outer(sample, hidden_probs) - np.outer(visible_sample, hidden_sample))
                self.visible_biases += self.lr * (sample - visible_sample)
                self.hidden_biases += self.lr * (hidden_probs - hidden_sample)

                # Update the total error
                total_error += np.mean((sample - visible_sample) ** 2)

            # Print the mean squared error for the current epoch
            print(f"Epoch {epoch + 1}/{self.epochs}, Mean Squared Error: {total_error / len(training_data)}")

            # Decay the learning rate
            if self.epoch_drop and (epoch + 1) % self.epoch_drop == 0:
                self.lr *= 1 - self.lr_decay

    def generate_sample(self):
        # Create a random initial visible state
        initial_state = np.random.randint(2, size=self.num_visible)

        # Sample the hidden layer based on the initial visible state
        hidden_sample = self.sample_hidden(initial_state)

        # Sample the visible layer based on the hidden layer sample
        generated_visible_sample = self.sample_visible(hidden_sample)

        return generated_visible_sample


def preprocess_data(data):
    data = data / 255.0
    data = (data > 0.5).astype(int)
    data = data.reshape(-1, 784)
    return data

def main():
    # Load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()

    # Create an instance of the ClassicalRBM class
    n_images = 10
    n_epochs = 10
    lr = 0.1
    num_visible = 784
    num_hidden = 20
    folder_path = 'results/2-tests/2-rbm/'

    # x_train = x_train_zero[0:n_images]
    x_train = x_train[0:n_images]

    training_data = preprocess_data(x_train)


    rbm = ClassicalRBM(num_visible, num_hidden, epochs=n_epochs, lr=lr)
    rbm.train(training_data)

    # Generate a new image using the generate_sample method
    new_image = rbm.generate_sample()

    # Display the generated image
    new_image = new_image.reshape(28, 28)
    plt.imshow(new_image, cmap='gray')
    plt.savefig(f'{folder_path}epochs_{n_epochs}-n_images_{n_images}-lr_{lr}.png')
    plt.close()


if __name__ == "__main__":
    main()
