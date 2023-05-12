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
        self.verbose = hyperparameters['training']['verbose']
        self.weights = np.random.normal(0, 0.1, (self.num_visible, self.num_hidden))
        self.visible_biases = np.zeros(self.num_visible)
        self.hidden_biases = np.zeros(self.num_hidden)
        self.counter = 0

        if self.qpu:
            self.sampler = EmbeddingComposite(DWaveSampler())
            self.sampler_name = self.sampler.child.solver.name
        else:
            self.sampler = SimulatedAnnealingSampler()
            self.sampler_name = "Simulated Annealing"


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

    def quantum_annealing(self, hamiltonian, num_reads=1, best_sol=False):
        model = hamiltonian.compile()
        bqm = model.to_bqm()

        sampleset = self.sampler.sample(bqm, num_reads=num_reads)
        
        if num_reads == 1 or best_sol:
            result_dict = sampleset.first.sample
            # Convert keys to integers and sort the dictionary by keys
            sorted_result = dict(sorted((int(key), value) for key, value in result_dict.items()))

            # Convert the sorted dictionary values to a list
            result_list = list(sorted_result.values())
            return result_list
        else: 
            solutions = []
            for sample in sampleset.record:
                solution = sample[0]
                num_occurrences = sample[2]
                solution_list = [(k, v) for k, v in enumerate(solution)]
                solution_list.sort(key=lambda tup: int(tup[0]))
                solution_list_final = [v for (k, v) in solution_list]

                for _ in range(num_occurrences):
                    solutions.append(solution_list_final)
        
            return solutions


    def sample_hidden(self, visible_vector, num_reads=1, best_sol=False):
        hamiltonian, hidden_variables = self.create_visible_hamiltonian(visible_vector, self.weights, self.hidden_biases)
        hidden_samples = self.quantum_annealing(hamiltonian, num_reads=num_reads, best_sol=best_sol)
        return hidden_samples

    def sample_visible(self, hidden_vector, num_reads=1, best_sol=False):
        hamiltonian, visible_variables = self.create_hidden_hamiltonian(self.visible_biases, self.weights, hidden_vector)
        visible_samples = self.quantum_annealing(hamiltonian, num_reads=num_reads, best_sol=best_sol)
        return visible_samples

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, training_data, num_reads=100):
        self.errors = []

        for epoch in range(self.epochs):
            # Shuffle the training data for each epoch
            np.random.shuffle(training_data)
            total_error = 0

            for sample in training_data:
                # Sample the hidden layer based on visible layer samples
                hidden_sample = self.sample_hidden(sample, num_reads=num_reads, best_sol=True)

                # Sample the visible layer based on hidden layer samples
                visible_sample = self.sample_visible(hidden_sample, num_reads=num_reads, best_sol=True)

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

        return sum(self.errors) / len(self.errors)


    def generate_sample(self):
        # Create a random initial visible state
        initial_state = np.random.randint(2, size=self.num_visible)

        # Sample the hidden layer based on the initial visible state
        hidden_sample = self.sample_hidden(initial_state, num_reads=10, best_sol=True)

        # Compute the probabilities for the generated visible layer
        generated_visible_probs = self.sigmoid(self.visible_biases + np.dot(hidden_sample, self.weights.T))

        # Threshold the probabilities to create a binary image
        generated_visible_sample = (generated_visible_probs > 0.5).astype(int)

        return generated_visible_sample

    def generate_samples(self, n_samples):
        # Create a random initial visible state
        initial_state = np.random.randint(2, size=self.num_visible)

        # Sample the hidden layer based on the initial visible state
        hidden_samples = self.sample_hidden(initial_state, num_reads=n_samples, best_sol=False)
    
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

    def generate_samples_1_by_1(self, n_samples):
        return tf.convert_to_tensor(np.array([self.generate_sample() for _ in range(n_samples)]))



def preprocess_mnist(data):
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

def generate_image(sample, hyperparameters, n_images):
    # Display the generated image
    new_image = sample.reshape(28, 28)
    plt.imshow(new_image, cmap='gray')
    
    model_type = hyperparameters['network']['qpu']
    folder_path = hyperparameters['plotting']['folder_path']
    
    file_name = f'{folder_path}qpu_{model_type}-epochs_{hyperparameters["training"]["epochs"]}-n_images_{n_images}-lr_{hyperparameters["training"]["lr"]}.png'
    
    plt.savefig(file_name)
    plt.close()

def main_mnist(hyperparameters, n_images):
    training_data = load_mnist(n_images, digits=[0  , 1])
    training_data = preprocess_mnist(training_data)

    qrbm = QuantumRBM(hyperparameters=hyperparameters)
    print('System Solver:', qrbm.sampler_name)
    qrbm.train(training_data)
    new_sample = qrbm.generate_sample()

    generate_image(new_sample, hyperparameters, n_images=n_images)


def discretize_data_distributions(data, num_bins):
    digitized_data = np.digitize(data, np.linspace(np.min(data), np.max(data), num_bins)) - 1
    binary_data = np.array([list(np.binary_repr(x, width=int(np.ceil(np.log2(num_bins))))) for x in digitized_data], dtype=int)
    return binary_data

def reconstruct_data_distributions(binary_data, num_bins, min_value, max_value):
    digitized_data = np.array([int("".join(map(str, row)), 2) for row in binary_data])
    continuous_data = np.linspace(min_value, max_value, num_bins)[digitized_data]
    return continuous_data

def plot_data_histogram(original_data, generated_data, num_bins):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].hist(original_data, bins=num_bins, density=True, alpha=0.75)
    ax[0].set_title("Original Gaussian Data")
    ax[0].set_ylabel("Probability Density")

    ax[1].hist(generated_data, bins=num_bins, density=True, alpha=0.75)
    ax[1].set_title("Generated Samples")
    ax[1].set_xlabel("Value")
    ax[1].set_ylabel("Probability Density")

    plt.tight_layout()
    plt.show()

def main_distributions(hyperparameters, num_samples):
    # Generate 1D Gaussian data
    data = np.random.normal(0, 1, num_samples)
    num_bins = 50

    # Discretize the data
    binary_data = discretize_data_distributions(data, num_bins)
    print('binary_data', binary_data[:5])

    # Train the QuantumRBM
    quantum_rbm = QuantumRBM(hyperparameters)
    quantum_rbm.train(binary_data)

    # Generate samples from the learned distribution
    generated_binary_samples = quantum_rbm.generate_samples_1_by_1(2).numpy()
    print('Generated Samples', generated_binary_samples[:2])

    # Reconstruct the continuous samples from the binary samples
    min_value = np.min(data)
    max_value = np.max(data)
    generated_samples = reconstruct_data_distributions(generated_binary_samples, num_bins, min_value, max_value)

    print("Generated samples:", generated_samples)

    # Plot the histograms
    plot_data_histogram(data, generated_samples, num_bins)

if __name__ == "__main__":
    mnist_main = True
    num_samples = 10
    num_bins = 50
    hyperparameters = {
        'network': {
            'num_visible': 784,
            'num_hidden': 20,
            'qpu': True,
        },
        'training': {
            'epochs': 10,
            'lr': 0.1,
            'verbose': True,
        },
        'plotting': {
            'folder_path': 'results/2-tests/d_quantum_rbm/'
        }
    }
    if mnist_main:
        main_mnist(hyperparameters=hyperparameters, n_images=num_samples)
    else:
        main_distributions(hyperparameters=hyperparameters, num_samples=num_samples)
