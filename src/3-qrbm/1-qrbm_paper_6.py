import numpy as np
from pyqubo import Binary
from dwave.system import EmbeddingComposite, DWaveSampler
#from dwave.system.samplers import SimulatedAnnealingSampler
from neal import SimulatedAnnealingSampler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class QuantumRBM:
    def __init__(self, num_visible, num_hidden, qpu=False, epochs=50, lr=0.1, lr_decay=0.1, epoch_drop=None, momentum=0, batch_size=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.qpu = qpu
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epoch_drop = epoch_drop
        self.momentum = momentum
        self.batch_size = batch_size
        self.weights = np.random.normal(0, 0.1, (num_visible, num_hidden))
        self.visible_biases = np.zeros(num_visible)
        self.hidden_biases = np.zeros(num_hidden)

    def create_visible_hamiltonian(self, visible_biases, weights, hidden_biases):
        hamiltonian = 0
        hidden_variables = [Binary(str(j)) for j in range(len(hidden_biases))]

        for i, visible_bias in enumerate(visible_biases):
            for j, hidden_bias in enumerate(hidden_biases):
                hamiltonian += -1 * weights[i][j] * hidden_variables[j]

        for j, hidden_bias in enumerate(hidden_biases):
            hamiltonian += -1 * hidden_bias * hidden_variables[j]

        return hamiltonian, hidden_variables

    def create_hidden_hamiltonian(self, visible_biases, weights, hidden_biases):
        hamiltonian = 0
        visible_variables = [Binary(str(j)) for j in range(len(visible_biases))]

        for i, hidden_bias in enumerate(hidden_biases):
            for j, visible_bias in enumerate(visible_biases):
                hamiltonian += -1 * weights[j][i] * visible_variables[j]

        for j, visible_bias in enumerate(visible_biases):
            hamiltonian += -1 * visible_bias * visible_variables[j]

        return hamiltonian, visible_variables

    def quantum_annealing(self, hamiltonian):
        model = hamiltonian.compile()
        bqm = model.to_bqm()

        if self.qpu:
            sampler = EmbeddingComposite(DWaveSampler())
        else:
            sampler = SimulatedAnnealingSampler()

        sampleset = sampler.sample(bqm, num_reads=1)
        solution = sampleset.first.sample

        solution_list = [(k, v) for k, v in solution.items()]
        solution_list.sort(key=lambda tup: int(tup[0]))
        solution_list_final = [v for (k, v) in solution_list]

        return solution_list_final

    def sample_hidden(self, visible_vector):
        hamiltonian, hidden_variables = self.create_visible_hamiltonian(visible_vector, self.weights, self.hidden_biases)
        hidden_sample = self.quantum_annealing(hamiltonian)
        return hidden_sample

    def sample_visible(self, hidden_vector):
        hamiltonian, visible_variables = self.create_hidden_hamiltonian(self.visible_biases, self.weights, hidden_vector)
        visible_sample = self.quantum_annealing(hamiltonian)
        return visible_sample

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, training_data, len_x=1, len_y=1):

        for epoch in range(self.epochs):
            # Shuffle the training data for each epoch
            np.random.shuffle(training_data)
            total_error = 0

            for sample in training_data:
                # Compute the probabilities for the hidden layer
                hidden_probs = self.sigmoid(self.hidden_biases + np.dot(sample, self.weights))
                
                # Sample the hidden layer based on visible layer sample
                hidden_sample = self.sample_hidden(sample)
                
                # Compute the probabilities for the visible layer
                visible_probs = self.sigmoid(self.visible_biases + np.dot(hidden_sample, self.weights.T))
                
                # Sample the visible layer based on hidden layer sample
                visible_sample = self.sample_visible(hidden_sample)

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

        # Compute the probabilities for the generated visible layer
        generated_visible_probs = self.sigmoid(self.visible_biases + np.dot(hidden_sample, self.weights.T))

        # Threshold the probabilities to create a binary image
        generated_visible_sample = (generated_visible_probs > 0.5).astype(int)

        return generated_visible_sample

def preprocess_data(data):
    data = data / 255.0
    data = (data > 0.5).astype(int)
    data = data.reshape(-1, 784)
    return data



def main():
    # Load MNIST dataset

    # Create an instance of the QuantumRBM class
    n_images = 2
    n_epochs = 2
    qpu = False 
    lr = 0.1
    num_visible = 784
    num_hidden = 20
    folder_path = 'results/2-tests/3-qrbm/'


    (x_train, y_train), (_, _) = mnist.load_data()

    # Filter images and labels for digit 0
    zero_indices = np.where((y_train == 0) | (y_train == 1))
    x_train_zero = x_train[zero_indices]
    y_train_zero = y_train[zero_indices]

    x_train = x_train_zero[0:n_images]


    # Preprocess the data
    training_data = preprocess_data(x_train)

    rbm = QuantumRBM(num_visible, num_hidden, epochs=n_epochs, lr=lr, lr_decay=0.1, epoch_drop=10, qpu=qpu)
    rbm.train(training_data)

    # Generate a new image using the generate_sample method
    generated_sample = rbm.generate_sample()

    # Display the generated image
    # Reshape the visible layer to a 2D image
    new_image = generated_sample.reshape(28, 28)
    plt.imshow(new_image, cmap='gray')
    plt.savefig(f'{folder_path}qpu_{qpu}-epochs_{n_epochs}-n_images_{n_images}-lr_{lr}.png')
    plt.close()


if __name__ == "__main__":
    main()
