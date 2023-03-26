import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
from classic_rbm import RBM



def preprocess_data(data_location):
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the pixel values to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Flatten the images from 28x28 to a 784-length vector
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)

    # Perform 1-hot encoding on the labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # Save the preprocessed data
    np.save(data_location+'train_images.npy', train_images)
    np.save(data_location+'train_labels.npy', train_labels)
    np.save(data_location+'test_images.npy', test_images)
    np.save(data_location+'test_labels.npy', test_labels)

    print('MNIST data preprocessed and saved.')


class QuantumRBM(RBM):
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, annealing_time=0.5):
        super().__init__(num_visible, num_hidden, learning_rate)
        self.annealing_time = annealing_time

        # Initialize the D-Wave quantum annealer
        self.sampler = EmbeddingComposite(DWaveSampler())

    def _build_qubo(self, visible_samples):
        # Calculate the contribution of the visible units to the hidden layer biases
        visible_contributions = visible_samples @ self.weights
        hidden_biases_including_visible = self.hidden_bias + visible_contributions

        qubos = []
        for sample_idx in range(visible_samples.shape[0]):
            # Create an empty QUBO matrix of size num_hidden x num_hidden
            qubo = np.zeros((self.num_hidden, self.num_hidden))

            # Fill the diagonal with the updated hidden biases for the current sample
            np.fill_diagonal(qubo, -hidden_biases_including_visible[sample_idx])

            # Fill the off-diagonal elements with the negative product of the corresponding weights
            for i in range(self.num_hidden):
                for j in range(i + 1, self.num_hidden):
                    qubo[i, j] = -self.weights[:, i].T @ self.weights[:, j]

            qubos.append({(i, j): qubo[i, j] for i in range(self.num_hidden) for j in range(self.num_hidden)})

        return qubos

    def _quantum_anneal(self, visible_samples):
        # Define the QUBO problem
        print('Building QUBO')
        qubos = self._build_qubo(visible_samples)
        print('Finished Building QUBO, sending to annealer')

        hidden_states = []
        for qubo in qubos:
            # Run the quantum annealing
            response = self.sampler.sample_qubo(qubo, num_reads=1, annealing_time=self.annealing_time)

            # Convert the results to binary hidden unit activations
            hidden_state = np.array([int(bit) for bit in response.first.sample.values()])
            hidden_states.append(hidden_state)

        print('Finished Annealer')
        return np.array(hidden_states)

    def train(self, data, epochs, batch_size):
        print('Training Quantum RBM...')

        num_samples = data.shape[0]

        for epoch in range(epochs):
            np.random.shuffle(data)

            for batch_start in range(0, num_samples, batch_size):
                batch = data[batch_start:batch_start + batch_size]

                print('Batch: ' + str(batch_start/batch_size))

                # Positive phase: Compute the hidden layer activations using quantum annealing
                hidden_states = self._quantum_anneal(batch)

                # Negative phase: Reconstruct the visible layer and update the weights
                visible_probs = self.sigmoid(hidden_states @ self.weights.T + self.visible_bias)
                hidden_probs_recon = self.sigmoid(visible_probs @ self.weights + self.hidden_bias)

                # Update weights and biases
                self.weights += self.learning_rate * (batch.T @ hidden_states - visible_probs.T @ hidden_probs_recon) / batch_size
                self.visible_bias += self.learning_rate * np.mean(batch - visible_probs, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(hidden_states - hidden_probs_recon, axis=0)

            print(f"Epoch {epoch + 1}/{epochs} completed.")

    def reconstruct(self, visible_samples):
        hidden_probs = self.sigmoid(visible_samples @ self.weights + self.hidden_bias)
        hidden_states = (hidden_probs > np.random.rand(*hidden_probs.shape)).astype(np.float32)
        visible_probs_recon = self.sigmoid(hidden_states @ self.weights.T + self.visible_bias)
        return visible_probs_recon

    def plot_generated_images(self):
        import matplotlib.pyplot as plt

        # Generate hidden states
        hidden_states = np.random.rand(100, self.num_hidden)

        # Compute visible probabilities and reshape them to images
        visible_probs = self.sigmoid(hidden_states @ self.weights.T + self.visible_bias)
        generated_images = visible_probs.reshape(-1, 28, 28)

        # Plot the generated images
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_images[i], cmap='gray')
            ax.axis('off')
        plt.show()




if __name__ == "__main__":

    data_location = 'data/mnist/'
    #preprocess_data(data_location)

    # Load the preprocessed MNIST data
    train_images = np.load(data_location + 'train_images.npy')
    train_labels = np.load(data_location + 'train_labels.npy')


    # Create and train the QuantumRBM
    quantum_rbm = QuantumRBM(num_visible=784, num_hidden=80, learning_rate=0.1)
    quantum_rbm.train(train_images, epochs=25, batch_size=100)

    # Reconstruct some images and calculate the reconstruction error
    reconstructed_images = quantum_rbm.reconstruct(train_images[:10])
    reconstruction_error = np.mean((train_images[:10] - reconstructed_images) ** 2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")

    # Generate and plot images after training
    quantum_rbm.plot_generated_images()