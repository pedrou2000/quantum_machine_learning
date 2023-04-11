import os
import json
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import norm, uniform, cauchy, pareto
import numpy as np
import matplotlib.pyplot as plt


class MMD_GAN:
    def __init__(self, z_dim, gen_hidden_units, critic_hidden_units, gen_lr, critic_lr, epochs, 
                 batch_size, update_ratio_critic, update_ratio_gen, mean, variance, results_path, 
                 print_frequency, mmd_lamb, sigmas, clip, target_dist, input_dist, plot_size, n_bins):
        self.z_dim = z_dim
        self.gen_hidden_units = gen_hidden_units
        self.critic_hidden_units = critic_hidden_units
        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.update_ratio_critic = update_ratio_critic
        self.update_ratio_gen = update_ratio_gen
        self.mean = mean
        self.variance = variance
        self.results_path = results_path
        self.print_frequency = print_frequency
        self.mmd_lamb = mmd_lamb
        self.sigmas = sigmas
        self.clip = clip
        self.target_dist = target_dist
        self.input_dist = input_dist
        self.plot_size = plot_size
        self.n_bins = n_bins

        self.generator_losses = []
        self.critic_losses = []

        self.generator = self.create_generator()
        self.critic = self.create_critic()

        self.generator_optimizer = Adam(learning_rate=self.gen_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
   
    def sample_real_data_batch(self):
        if self.target_dist == "gaussian":
            return np.random.normal(self.mean, np.sqrt(self.variance), (self.batch_size, 1)).astype(np.float32)
        elif self.target_dist == "uniform":
            return np.random.uniform(self.mean, self.variance, (self.batch_size, 1)).astype(np.float32)
        elif self.target_dist == "cauchy":
            return cauchy.rvs(self.mean,  np.sqrt(self.variance), size=(self.batch_size, 1)).astype(np.float32)
        elif self.target_dist == "pareto":
            return pareto.rvs(self.mean, scale=np.sqrt(self.variance), size=(self.batch_size, 1)).astype(np.float32)

    def sample_noise(self, plot=False):
        if not plot: 
            if self.input_dist == "gaussian":
                return np.random.normal(0, 1, (self.batch_size, self.z_dim)).astype(np.float32)
            elif self.input_dist == "uniform":
                return np.random.uniform(0, 1,(self.batch_size, self.z_dim)).astype(np.float32)
        else:
            if self.input_dist == "gaussian":
                return np.random.normal(0, 1, (self.plot_size, self.z_dim)).astype(np.float32)
            elif self.input_dist == "uniform":
                return np.random.uniform(0, 1,(self.plot_size, self.z_dim)).astype(np.float32)

    def create_generator(self):
        model = Sequential()
        model.add(Dense(self.gen_hidden_units[0], input_dim=self.z_dim))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.gen_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(1))
        return model

    def create_critic(self):
        model = Sequential()
        model.add(Dense(self.critic_hidden_units[0], input_dim=1))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[1]))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[2]))
        model.add(ELU())
        model.add(Dense(1))
        return model

    def gaussian_kernel_matrix(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        x = tf.reshape(x, (x_size, 1, dim))
        y = tf.reshape(y, (1, y_size, dim))
        tile = tf.tile(x, (1, y_size, 1))
        diff = tile - y

        sum_squares = tf.reduce_sum(tf.square(diff), axis=-1)
        sum_squares = tf.expand_dims(sum_squares, axis=-1)

        beta = 1.0 / (2.0 * tf.cast(self.sigmas, tf.float32))
        exponent = -beta * sum_squares

        return tf.reduce_sum(tf.exp(exponent), axis=-1)


    def compute_mmd_mean(self, x_real, x_fake):
        xx = self.gaussian_kernel_matrix(x_real, x_real)
        yy = self.gaussian_kernel_matrix(x_fake, x_fake)
        xy = self.gaussian_kernel_matrix(x_real, x_fake)
        mmd = tf.reduce_mean(xx) - 2 * tf.reduce_mean(xy) + tf.reduce_mean(yy)
        return mmd

    def compute_mmd(self, x_real, x_fake):
        xx = self.gaussian_kernel_matrix(x_real, x_real)
        yy = self.gaussian_kernel_matrix(x_fake, x_fake)
        xy = self.gaussian_kernel_matrix(x_real, x_fake)
        n = tf.cast(tf.shape(x_real)[0], dtype=tf.float32)
        mmd = (tf.reduce_sum(xx) - tf.linalg.trace(xx)) / (n * (n - 1))
        mmd -= 2 * tf.reduce_sum(xy) / (n * n)
        mmd += (tf.reduce_sum(yy) - tf.linalg.trace(yy)) / (n * (n - 1))
        return mmd

    def train(self):
        average_critic_loss = 0
        average_gen_loss = 0

        for epoch in range(self.epochs):

            # Critic Training Step
            for _ in range(self.update_ratio_critic):
                x_real = self.sample_real_data_batch()
                z = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake = self.generator(z)
                    critic_x_real = self.critic(tf.expand_dims(x_real, -1))
                    critic_x_fake = self.critic(x_fake)
                    mmd = self.compute_mmd(critic_x_real, critic_x_fake)
                    constraint_term = tf.reduce_mean(critic_x_real) - tf.reduce_mean(critic_x_fake)
                    critic_loss = -mmd + self.mmd_lamb * tf.minimum(constraint_term, 0)

                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                average_critic_loss += critic_loss

                # Clip critic weights
                for var in self.critic.trainable_variables:
                    var.assign(tf.clip_by_value(var, -self.clip, self.clip))

            # Generator Training Step
            for _ in range(self.update_ratio_gen):
                z = self.sample_noise()

                with tf.GradientTape() as tape:
                    x_fake = self.generator(z)
                    critic_x_fake = self.critic(x_fake)
                    mmd = self.compute_mmd(critic_x_real, critic_x_fake)
                    generator_loss = mmd
                generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                average_gen_loss += generator_loss

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
        z1 = self.sample_noise(plot=True)
        x_fake = self.generator(z1).numpy().flatten()

        plt.hist(x_fake, bins=self.n_bins, alpha=0.6, label="Generated Data", density=True) # set density=True to display percentages

        # Plot PDFs
        x_values = np.linspace(self.mean - 4 * np.sqrt(self.variance), self.mean + 4 * np.sqrt(self.variance), 1000)

        if self.target_dist == "gaussian":
            pdf_values = norm.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "uniform":
            lower_bound = self.mean
            upper_bound = self.variance
            scale = upper_bound - lower_bound
            pdf_values = uniform.pdf(x_values, loc=self.mean, scale=scale)
        elif self.target_dist == "cauchy":
            pdf_values = cauchy.pdf(x_values, loc=self.mean, scale=np.sqrt(self.variance))
        elif self.target_dist == "pareto":
            pdf_values = pareto.pdf(x_values, b=self.mean, scale=np.sqrt(self.variance))

        pdf_values = pdf_values / (pdf_values.sum() * np.diff(x_values)[0]) # normalize the PDF

        plt.plot(x_values, pdf_values, label="Real PDF")

        plt.legend()

        plt.savefig(f"{folder_path}histogram.png", dpi=300)
        plt.close()



    def plot_losses(self, folder_path):
        plt.plot(self.critic_losses, label='Critic MMD')
        plt.plot(self.generator_losses, label='Generator MMD')
        plt.xlabel('Epoch x '+str(self.print_frequency))
        plt.ylabel('MMD')
        plt.legend()
        plt.title('MMDs')

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
            "mean": self.mean,
            "variance": self.variance,
            "results_path": self.results_path,
            "print_frequency": self.print_frequency,
            "mmd_lamb": self.mmd_lamb,
            "sigmas": self.sigmas,
            "clip": self.clip,
            "target_dist": self.target_dist,
            "input_dist": self.input_dist,
            "plot_size": self.plot_size,
            "n_bins": self.n_bins,
        }
        with open(os.path.join(folder_path, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=4)
    
    def create_result_folder(self):
        timestamp = time.strftime("%m%d_%H%M")
        folder_path = (
            self.results_path + f"epochs_{self.epochs}_target_{self.target_dist}_input_{self.input_dist}_{self.variance}_{timestamp}/"
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
    epochs = 10000
    print_frequency = 100
    z_dim = 2
    gen_hidden_units = [7, 13, 7]
    critic_hidden_units = [11, 29, 11]
    lr = 1e-3
    batch_size = 64
    update_ratio_critic = 2
    update_ratio_gen = 1
    mean = 23
    variance = 1
    results_path = "results/gan/1d/mmd/setwise_paper_4/10000/try_3/"
    mmd_lamb = 0.01
    clip = 1
    sigmas = [1,2,4,8,16]
    target_dists = ["gaussian"] 
    input_dists = ["gaussian", "uniform"]
    plot_size = 10000
    n_bins = 100


    for input_dist in input_dists:
        for target_dist in target_dists:
                mmd_gan = MMD_GAN(
                    z_dim=z_dim,
                    gen_hidden_units=gen_hidden_units,
                    critic_hidden_units=critic_hidden_units,
                    gen_lr=lr,
                    critic_lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    update_ratio_critic=update_ratio_critic,
                    update_ratio_gen=update_ratio_gen,
                    mean=mean,
                    variance=variance,
                    results_path=results_path,
                    print_frequency=print_frequency,
                    mmd_lamb=mmd_lamb,
                    sigmas=sigmas,
                    clip=clip,
                    target_dist=target_dist,
                    input_dist=input_dist,
                    plot_size=plot_size,
                    n_bins=n_bins,
                )

                mmd_gan.train()
                mmd_gan.plot_and_save()