import numpy as np
from pyqubo import Binary
from dwave.system import EmbeddingComposite, DWaveSampler
#from dwave.system.samplers import SimulatedAnnealingSampler
from neal import SimulatedAnnealingSampler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import concurrent.futures


class QuantumRBM:
    def __init__(self, hyperparameters):
        self.num_visible = hyperparameters['network']['num_visible']
        self.num_hidden = hyperparameters['network']['num_hidden']
        self.qpu = hyperparameters['network']['qpu']
        self.epochs = hyperparameters['training']['epochs']
        self.lr = hyperparameters['training']['lr']
        self.lr_decay = hyperparameters['training']['lr_decay']
        self.epoch_drop = hyperparameters['training']['epoch_drop']
        self.momentum = hyperparameters['training']['momentum']
        self.batch_size = hyperparameters['training']['batch_size']
        self.verbose = hyperparameters['training']['verbose']
        self.weights = np.random.normal(0, 0.1, (self.num_visible, self.num_hidden))
        self.visible_biases = np.zeros(self.num_visible)
        self.hidden_biases = np.zeros(self.num_hidden)

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

    def quantum_annealing(self, hamiltonian, num_reads=1):
        model = hamiltonian.compile()
        bqm = model.to_bqm()

        if self.qpu:
            sampler = EmbeddingComposite(DWaveSampler())
        else:
            sampler = SimulatedAnnealingSampler()

        sampleset = sampler.sample(bqm, num_reads=num_reads)
        solutions = []

        for sample in sampleset.record:
            solution = sample[0]
            num_occurrences = sample[2]
            solution_list = [(k, v) for k, v in enumerate(solution)]
            solution_list.sort(key=lambda tup: int(tup[0]))
            solution_list_final = [v for (k, v) in solution_list]

            for _ in range(num_occurrences):
                solutions.append(solution_list_final)

        if len(solutions) == 1:
            return solutions[0]
        else:
            return solutions


    def sample_hidden(self, visible_vector, num_reads=1):
        hamiltonian, hidden_variables = self.create_visible_hamiltonian(visible_vector, self.weights, self.hidden_biases)
        hidden_samples = self.quantum_annealing(hamiltonian, num_reads=num_reads)
        return hidden_samples

    def sample_visible(self, hidden_vector, num_reads=1):
        hamiltonian, visible_variables = self.create_hidden_hamiltonian(self.visible_biases, self.weights, hidden_vector)
        visible_samples = self.quantum_annealing(hamiltonian, num_reads=num_reads)
        return visible_samples

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, training_data, num_reads=1):
        self.errors = []

        for epoch in range(self.epochs):
            # Shuffle the training data for each epoch
            np.random.shuffle(training_data)
            total_error = 0

            for sample in training_data:
                # Sample the hidden layer based on visible layer samples
                hidden_samples = self.sample_hidden(sample, num_reads=num_reads)
                hidden_sample = np.mean(hidden_samples, axis=0)

                # Sample the visible layer based on hidden layer samples
                visible_samples = self.sample_visible(hidden_sample, num_reads=num_reads)
                visible_sample = np.mean(visible_samples, axis=0)

                # Compute the probabilities for the hidden layer
                hidden_probs = self.sigmoid(self.hidden_biases + np.dot(sample, self.weights))

                # Compute the probabilities for the visible layer
                visible_probs = self.sigmoid(self.visible_biases + np.dot(hidden_sample, self.weights.T))

                # Update the weights and biases
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

            # Decay the learning rate
            if self.epoch_drop and (epoch + 1) % self.epoch_drop == 0:
                self.lr *= 1 - self.lr_decay

        return sum(self.errors) / len(self.errors)


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

    def generate_samples(self, n_samples):
        # Create a random initial visible state
        initial_state = np.random.randint(2, size=self.num_visible)

        # Sample the hidden layer based on the initial visible state
        hidden_samples = self.sample_hidden(initial_state, num_reads=n_samples)
    
        generated_visible_samples = []

        for hidden_sample in hidden_samples:
            # Compute the probabilities for the generated visible layer
            generated_visible_probs = self.sigmoid(self.visible_biases + np.dot(hidden_sample, self.weights.T))

            # Threshold the probabilities to create a binary image
            generated_visible_sample = (generated_visible_probs > 0.5).astype(int)
            generated_visible_samples.append(generated_visible_sample)

        samples_array = np.array(generated_visible_samples)
        samples_tensor = tf.convert_to_tensor(samples_array)

        return samples_tensor



def preprocess_data(data):
    data = data / 255.0
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
    
    model_type = hyperparameters['network']['qpu']
    folder_path = hyperparameters['plotting']['folder_path']
    
    file_name = f'{folder_path}qpu_{model_type}-epochs_{hyperparameters["training"]["epochs"]}-n_images_{hyperparameters["training"]["n_images"]}-lr_{hyperparameters["training"]["lr"]}.png'
    
    plt.savefig(file_name)
    plt.close()


def main(hyperparameters):
    training_data = load_mnist(hyperparameters['training']['n_images'], digits=[0, 1])
    training_data = preprocess_data(training_data)

    qrbm = QuantumRBM(hyperparameters=hyperparameters)
    qrbm.train(training_data)
    new_sample = qrbm.generate_sample()

    generate_image(new_sample, hyperparameters)

if __name__ == "__main__":
    hyperparameters = {
        'network': {
            'num_visible': 784,
            'num_hidden': 20,
            'qpu': True,
        },
        'training': {
            'epochs': 2,
            'lr': 0.1,
            'lr_decay': 0.1,
            'epoch_drop': 10,
            'momentum': 0,
            'batch_size': None,
            'n_images': 10,
            'verbose': True,
        },
        'plotting': {
            'folder_path': 'results/2-tests/d_quantum_rbm/'
        }
    }
    main(hyperparameters=hyperparameters)
