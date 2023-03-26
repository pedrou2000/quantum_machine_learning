import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import time

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate, num_epochs, batch_size, k, num_samples, target_mean, target_sd, target_normal, results_path):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        self.target_mean = target_mean
        self.target_sd = target_sd
        self.target_normal = target_normal
        self.results_path = results_path

        self.W = tf.Variable(tf.random.normal([n_visible, n_hidden], mean=0.0, stddev=0.01, dtype=tf.float64), dtype=tf.float64)
        self.b_v = tf.Variable(tf.zeros([1, n_visible], dtype=tf.float64), dtype=tf.float64)
        self.b_h = tf.Variable(tf.zeros([1, n_hidden], dtype=tf.float64), dtype=tf.float64)

        self.train_data = self.generate_data()

    def generate_data(self):
        if self.target_normal:
            return np.random.normal(self.target_mean, self.target_sd, self.num_samples).astype(np.float64).reshape(-1, 1)
        else:
            return np.random.uniform(self.target_mean, self.target_sd, self.num_samples).astype(np.float64).reshape(-1, 1)


    def sample(self, probabilities):
        return tf.cast(tf.random.uniform(tf.shape(probabilities), dtype=probabilities.dtype) < probabilities, dtype=tf.float64)
 
    def train(self):
        num_batches = self.train_data.shape[0] // self.batch_size
        self.reconstruction_errors = []
        save_every_n = 10
        epoch_error_total = 0

        for epoch in range(self.num_epochs):
            np.random.shuffle(self.train_data)
            epoch_error = 0

            for i in range(num_batches):
                v0_state = self.train_data[i * self.batch_size:(i + 1) * self.batch_size]
                h0_prob = tf.sigmoid(tf.matmul(v0_state, self.W) + self.b_h)
                h0_state = self.sample(h0_prob)

                v_state = v0_state
                h_state = h0_state

                for _ in range(self.k):
                    v_state = tf.matmul(h_state, self.W, transpose_b=True) + self.b_v  # Use identity activation function for visible units
                    h_prob = tf.sigmoid(tf.matmul(v_state, self.W) + self.b_h)
                    h_state = self.sample(h_prob)

                self.W.assign_add(self.learning_rate * (tf.matmul(v0_state, h0_prob, transpose_a=True) - tf.matmul(v_state, h_prob, transpose_a=True)) / self.batch_size)
                self.b_v.assign_add(self.learning_rate * tf.reduce_sum(v0_state - v_state, axis=0, keepdims=True))
                self.b_h.assign_add(self.learning_rate * tf.reduce_mean(h0_prob - h_prob, axis=0, keepdims=True))

                error = tf.reduce_mean(tf.square(v0_state - v_state))
                epoch_error += error

            epoch_error /= num_batches
            epoch_error_total += epoch_error

            if epoch % save_every_n == 0:
                epoch_error_total /= save_every_n
                print(f"Epoch {epoch}, reconstruction error: {epoch_error_total}")
                self.reconstruction_errors.append(epoch_error_total)
                epoch_error_total = 0

    def generate_samples(self):
        samples = []
        v = tf.zeros((1, self.n_visible), dtype=tf.float64)

        for _ in range(self.num_samples):
            h_prob = tf.sigmoid(tf.matmul(v, self.W) + self.b_h)
            h = self.sample(h_prob)
            v = tf.matmul(h, tf.transpose(self.W)) + self.b_v  # Use identity activation function for visible units
            samples.append(v.numpy()[0])

        return np.array(samples)

    
    def plot_losses(self, folder_path):
        plt.plot(self.reconstruction_errors, label='Reconstruction Error')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.title('Losses')

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def plot_results(self, folder_path):
        generated_samples = self.generate_samples()
        plt.hist(generated_samples, bins=50, alpha=0.6, label='Generated Data')
        plt.hist(self.train_data, bins=50, alpha=0.6, label='Real Data')
        plt.legend()

        plt.savefig(f'{folder_path}histogram.png', dpi=300)
        plt.close()

    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'k': self.k,
            'target_mean': self.target_mean,
            'target_sd': self.target_sd,
            'target_normal': self.target_normal,
            'results_path': self.results_path,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
    
    def create_result_folder(self):
        distribution_type = 'normal' if self.target_normal else 'uniform'
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = self.results_path + f'{distribution_type}_{self.target_mean}_{self.target_sd}_{timestamp}/'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def plot_and_save(self):
        folder_path = self.create_result_folder()
        self.plot_results(folder_path)
        self.plot_losses(folder_path)  # Add this line to call the plot_losses function
        self.save_parameters_to_json(folder_path)





def main(n_visible, n_hidden, learning_rate, num_epochs, batch_size, k, num_samples, target_mean, target_sd, target_normal, results_path):
    rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=learning_rate, num_epochs=num_epochs, 
              batch_size=batch_size, k=k, num_samples=num_samples, target_mean=target_mean, target_sd=target_sd, 
              target_normal=target_normal, results_path=results_path)
    rbm.train()
    rbm.plot_and_save()


if __name__ == "__main__":
    # Parameters
    num_epochs = 1000
    n_visible = 1
    n_hidden = 100
    learning_rate = 0.01
    batch_size = 128
    k = 3
    num_samples = 10000
    results_path = 'results/rbm/1d/'
    target_means = [0, 5, 10]
    target_sds = [0.1, 0.2, 0.4, 0.8]

    for target_normal in [False]:
        for target_mean in target_means:
            for target_sd in target_sds:
                distribution_type = 'normal' if target_normal else 'uniform'
                print(f"Running RBM for {distribution_type} distribution with mean: {target_mean}, sd: {target_sd}")
                main(n_visible, n_hidden, learning_rate, num_epochs, batch_size, k, num_samples, target_mean, target_sd, target_normal, results_path)
