import os
import json
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


class MMD_GAN:
    def __init__(self, z_dim=1, gen_hidden_units=[7, 13, 7], critic_hidden_units=[11, 29, 11],
        gen_lr=1e-3, critic_lr=1e-3, epochs=300, batch_size=64,update_ratio_critic=4, update_ratio_gen=4, 
        n=100, mean=0, variance=1, results_path="results/gan/1d/mmd/", print_frequency=10):

        self.z_dim = z_dim
        self.gen_hidden_units = gen_hidden_units
        self.critic_hidden_units = critic_hidden_units
        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.update_ratio_critic = update_ratio_critic
        self.update_ratio_gen = update_ratio_gen
        self.n = n
        self.mean = mean
        self.variance = variance
        self.results_path = results_path
        self.print_frequency = print_frequency
        self.generator_losses = []
        self.critic_losses = []

        self.generator = self.create_generator()
        self.critic = self.create_critic()
       
        self.generator_optimizer = Adam(learning_rate=self.gen_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

    def sample_real_data_batch(self):
        return np.random.normal(self.mean, np.sqrt(self.variance), (self.batch_size, self.n, 1)).astype(np.float32)

    def sample_noise(self):
        return tf.random.normal((self.batch_size, self.z_dim * self.n))

    def create_generator(self):
        model = Sequential()
        model.add(Dense(self.gen_hidden_units[0], input_dim=self.z_dim * self.n))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(self.n))
        return model

    def create_critic(self):
        model = Sequential()
        model.add(Dense(self.critic_hidden_units[0], input_dim=self.z_dim * self.n))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(1))
        return model

    def gaussian_kernel(self, x, y, sigma=1.0):
        return tf.exp(-tf.reduce_sum((x - y) ** 2, axis=-1) / (2 * sigma ** 2))

    def compute_mmd(self, critic_x_real1, critic_x_real2, critic_x_fake1, critic_x_fake2):
        k_xx = self.gaussian_kernel(critic_x_real1, critic_x_real2)
        k_xy = self.gaussian_kernel(critic_x_real1, critic_x_fake1)
        k_yy = self.gaussian_kernel(critic_x_fake1, critic_x_fake2)
        mmd = tf.reduce_mean(k_xx) - 2 * tf.reduce_mean(k_xy) + tf.reduce_mean(k_yy)
        return mmd

    def train(self):
        average_critic_loss = 0
        average_critic_loss_aux = 0
        average_gen_loss = 0
        average_gen_loss_aux = 0

        for epoch in range(self.epochs):

            # Critic Training Step
            for _ in range(self.update_ratio_critic):
                x_real1 = self.sample_real_data_batch()
                x_real2 = self.sample_real_data_batch()
                z1 = self.sample_noise()
                z2 = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake1 = self.generator(z1)
                    x_fake2 = self.generator(z2)
                    critic_real1 = self.critic(x_real1)
                    critic_real2 = self.critic(x_real2)
                    critic_fake1 = self.critic(x_fake1)
                    critic_fake2 = self.critic(x_fake2)
                    mmd = self.compute_mmd(critic_real1, critic_real2, critic_fake1, critic_fake2)
                    critic_loss = -mmd

                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                average_critic_loss_aux += critic_loss

            average_critic_loss += average_critic_loss_aux / self.update_ratio_critic
            average_critic_loss_aux = 0

            # Generator training step
            for _ in range(self.update_ratio_gen):
                z1 = self.sample_noise()
                z2 = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake1 = self.generator(z1)
                    x_fake2 = self.generator(z2)
                    critic_fake1 = self.critic(x_fake1)
                    critic_fake2 = self.critic(x_fake2)
                    mmd = self.compute_mmd(critic_real1, critic_real2, critic_fake1, critic_fake2)
                    generator_loss = mmd

                generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                average_gen_loss_aux += generator_loss
            
            average_gen_loss += average_critic_loss_aux / self.update_ratio_gen
            average_gen_loss_aux = 0
            
            # Registering Results
            if epoch % self.print_frequency == 0:
                average_gen_loss /= self.print_frequency
                average_critic_loss /= self.print_frequency
                self.generator_losses.append(average_gen_loss)
                self.critic_losses.append(average_critic_loss)
                print(f"Epoch {epoch}: generator MMD = {average_gen_loss:.4f}, critic MMD = {average_critic_loss:.4f}")
                average_gen_loss = 0
                average_critic_loss = 0
    
    def plot_results(self, folder_path):
        z1 = self.sample_noise()
        x_fake = self.generator(z1).numpy()[0]
        x_real = self.sample_real_data_batch()[0]
        x_real = [x[0] for x in x_real]

        plt.hist(x_fake, bins=50, alpha=0.6, label="Generated Data")
        plt.hist(x_real, bins=50, alpha=0.6, label="Real Data")
        plt.legend()

        plt.savefig(f"{folder_path}histogram.png", dpi=300)
        plt.close()

    def plot_losses(self, folder_path):
        plt.plot(self.critic_losses, label='Critic Loss')
        plt.plot(self.generator_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses')

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def save_parameters_to_json(self, folder_path):
        parameters = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "z_dim": self.z_dim,
            "gen_hidden_units": self.gen_hidden_units,
            "critic_hidden_units": self.critic_hidden_units,
            "gen_lr": self.gen_lr,
            "critic_lr": self.critic_lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "update_ratio_critic": self.update_ratio_critic,
            "update_ratio_gen": self.update_ratio_gen,
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
        }
        with open(os.path.join(folder_path, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=4)

    def create_result_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_path = (
            self.results_path + f"mmd_gan_{self.mean}_{self.variance}_{timestamp}/"
        )
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def plot_and_save(self):
        folder_path = self.create_result_folder()
        self.plot_results(folder_path)
        self.plot_losses(folder_path)
        self.save_parameters_to_json(folder_path)


if __name__ == "__main__":

    # Hyperparameters
    epochs = 1000
    print_frequency = 10
    z_dim = 1
    gen_hidden_units = [7, 13, 7]
    critic_hidden_units = [11, 29, 11]
    gen_lr = 1e-3
    critic_lr = 1e-3
    batch_size = 64
    update_ratio_critic = 1
    update_ratio_gen = 1
    n = 100
    means = [0, ]
    variances = [8]
    results_path = "results/gan/1d/mmd/"


    for mean in means:
        for variance in variances:
            print(f"Running GAN for normal distribution with mean: {mean}, variance: {variance}")

            mmd_gan = MMD_GAN(
                z_dim=z_dim, gen_hidden_units=gen_hidden_units, critic_hidden_units=critic_hidden_units,
                gen_lr=gen_lr, critic_lr=critic_lr, epochs=epochs, batch_size=batch_size,
                update_ratio_critic=update_ratio_critic, update_ratio_gen=update_ratio_gen, 
                n=n, mean=mean, variance=variance, results_path=results_path, print_frequency=print_frequency,
            )

            mmd_gan.train()
            mmd_gan.plot_and_save()