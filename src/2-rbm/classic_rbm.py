import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        
        # Initialize the weight matrix and biases
        self.weights = np.random.normal(0, 0.1, (num_visible, num_hidden))
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, data, epochs, batch_size):
        num_samples = data.shape[0]
        
        for epoch in range(epochs):
            np.random.shuffle(data)
            
            for batch_start in range(0, num_samples, batch_size):
                batch = data[batch_start:batch_start + batch_size]
                
                # Positive phase: Compute the hidden layer activations
                hidden_probs = self.sigmoid(batch @ self.weights + self.hidden_bias)
                hidden_states = (hidden_probs > np.random.rand(*hidden_probs.shape)).astype(int)
                
                # Negative phase: Reconstruct the visible layer and update the weights
                visible_probs = self.sigmoid(hidden_states @ self.weights.T + self.visible_bias)
                hidden_probs_recon = self.sigmoid(visible_probs @ self.weights + self.hidden_bias)
                
                # Update weights and biases
                self.weights += self.learning_rate * (batch.T @ hidden_probs - visible_probs.T @ hidden_probs_recon) / batch_size
                self.visible_bias += self.learning_rate * np.mean(batch - visible_probs, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(hidden_probs - hidden_probs_recon, axis=0)
            
            print(f"Epoch {epoch + 1}/{epochs} completed.")
    
    def reconstruct(self, data):
        hidden_probs = self.sigmoid(data @ self.weights + self.hidden_bias)
        visible_probs = self.sigmoid(hidden_probs @ self.weights.T + self.visible_bias)
        return visible_probs

    def generate_images(self, num_samples=10, num_gibbs_steps=1000):
        # Initialize visible units randomly
        visible_samples = np.random.binomial(1, 0.5, (num_samples, self.num_visible))

        # Perform Gibbs sampling
        for _ in range(num_gibbs_steps):
            hidden_probs = self.sigmoid(visible_samples @ self.weights + self.hidden_bias)
            hidden_samples = np.random.binomial(1, hidden_probs)

            visible_probs = self.sigmoid(hidden_samples @ self.weights.T + self.visible_bias)
            visible_samples = np.random.binomial(1, visible_probs)

        return visible_samples

    def plot_generated_images(self, num_samples=10, num_gibbs_steps=1000):
        generated_images = self.generate_images(num_samples, num_gibbs_steps)

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
        axes = axes.ravel()

        for i, img in enumerate(generated_images):
            axes[i].imshow(img.reshape(28, 28), cmap='gray')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    data_location = 'data/mnist/'
    preprocess_data(data_location)


    # Load the preprocessed MNIST data
    train_images = np.load(data_location+'train_images.npy')
    train_labels = np.load(data_location+'train_labels.npy')

    # Create and train the RBM
    rbm = RBM(num_visible=784, num_hidden=80, learning_rate=0.1)
    rbm.train(train_images, epochs=25, batch_size=100)

    # Reconstruct some images and calculate the reconstruction error
    reconstructed_images = rbm.reconstruct(train_images[:10])
    reconstruction_error = np.mean((train_images[:10] - reconstructed_images) ** 2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")


    # Generate and plot images after training
    rbm.plot_generated_images()


