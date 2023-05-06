import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time
import json
from scipy.stats import norm, uniform, cauchy, pareto, beta
from a_vanilla_gan_1d import *
from b_mmd_gan_1d import *
from c_classical_rbm import *
from d_quantum_rbm import *

class MMD_QAAAN(MMD_GAN):
    def __init__(self, hyperparameters):
        self.feature_layer_size = hyperparameters['hyperparameters_qaaan']['network']['feature_layer_size']
        self.update_ratios = hyperparameters['hyperparameters_qaaan']['training']['update_ratios']
        self.rbm_type = hyperparameters['hyperparameters_qaaan']['network']['rbm_type']
        self.train_rbm_every_n = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_every_n']
        self.samples_train_rbm = hyperparameters['hyperparameters_qaaan']['training']['samples_train_rbm']
        self.train_rbm_cutoff_epoch = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_cutoff_epoch']
        self.train_rbm_start_epoch = hyperparameters['hyperparameters_qaaan']['training']['train_rbm_start_epoch']

        super().__init__(hyperparameters['hyperparameters_mmd_gan'])
        self.hyperparameters = hyperparameters

        # Create a new model to extract intermediate layer output
        self.feature_layer_model = self.create_feature_layer_model(layer_index=-2)
        self.rbm_losses = []

        # Create the Restricted Boltzmann Machine which will work as Prior to the Generator
        if self.rbm_type == 'classical':
            self.rbm = ClassicalRBM(hyperparameters=hyperparameters['hyperparameters_rbm'])
        elif self.rbm_type == 'simulated' or self.rbm_type == 'quantum':
            self.rbm = QuantumRBM(hyperparameters=hyperparameters['hyperparameters_rbm'])

    def create_critic(self):
        model = Sequential()
        model.add(Dense(self.critic_hidden_units[0], input_dim=1))
        model.add(ELU())
        model.add(Dense(self.critic_hidden_units[1]))
        model.add(ELU())
        model.add(layers.Dense(self.feature_layer_size, activation='tanh'))
        model.add(Dense(1))
        return model

    def create_feature_layer_model(self, layer_index):
        input_layer = self.critic.input
        intermediate_layer = self.critic.get_layer(index=layer_index).output
        return keras.Model(inputs=input_layer, outputs=intermediate_layer)
    
    def preprocess_rbm_input(self, data):
        data = (data > 0).astype(int)
        return data
    
    def reparameterize_vector(self, z, reparam_type='paper', alpha=1, beta_alpha=0.5, beta_beta=0.5):
        zeta = np.zeros_like(z, dtype=float)
        u = np.random.uniform(0, 1, size=z.shape)

        if reparam_type == 'paper':
            idx = z == 1
            zeta[idx] = 1 - (1 / alpha) * np.log(1 - (1 - np.exp(-2 * alpha)) * u[idx])
            zeta[~idx] = -1
        elif reparam_type == 'beta':
            mapped_samples = (z + 1) / 2
            reparameterized_samples = beta.ppf(u, beta_alpha, beta_beta)
            zeta = reparameterized_samples * 2 - 1
        else:
            raise ValueError("Invalid reparam_type. Accepted values are 'current' and 'beta'")

        return zeta
        
    def generate_prior(self, n_samples=None, n_batches=None, n_lists=None):
        if n_samples is None:
            if n_batches is None:
                total_samples = self.batch_size
            else:
                total_samples = self.batch_size * n_batches
        else:
            total_samples = n_samples

        if n_lists is not None:
            total_samples *= n_lists

        # Generate samples in chunks of max_sampling and concatenate them
        max_sampling = 500
        chunks = total_samples // max_sampling
        remainder = total_samples % max_sampling

        if chunks > 0:
            samples = self.rbm.generate_samples(max_sampling)
            rbm_prior_chunks = [samples for _ in range(chunks)]
            if remainder > 0:
                rbm_prior_chunks.append(samples[:remainder])
        else:
            samples = self.rbm.generate_samples(total_samples)
            rbm_prior_chunks = [samples]

        rbm_prior = np.concatenate(rbm_prior_chunks)
        rbm_prior = self.reparameterize_vector(rbm_prior, reparam_type='paper', alpha=1)

        # If n_batches and n_lists are specified, reshape the samples into a list of lists of batches
        if n_batches is not None and n_lists is not None:
            list_of_batches = [
                [
                    rbm_prior[k * self.batch_size:(k + 1) * self.batch_size]
                    for k in range(i * n_batches, (i + 1) * n_batches)
                ]
                for i in range(n_lists)
            ]
            return list_of_batches
        elif n_batches is not None:
            batches = [rbm_prior[i * self.batch_size:(i + 1) * self.batch_size] for i in range(n_batches)]
            return batches
        else:
            return rbm_prior

    def train(self):
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0
        rbm_loss_total = 0
        rbm_loss = 0
        average_critic_loss = 0
        average_gen_loss = 0

        rbm_prior_critic = self.generate_prior(n_batches=self.update_ratios['critic'])

        for epoch in range(self.epochs):

            # Critic Training
            # for j in range(self.update_ratios['critic']):
            #     # Train critic on real data
            #     real_data = self.sample_real_data()
            #     d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

            #     # Train critic on generated data
            #     rbm_prior = rbm_prior_critic[j]
            #     generated_data = self.generator.predict(rbm_prior, verbose=0)
            #     d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # Critic Training 
            for j in range(self.update_ratios['critic']):
                x_real = self.sample_real_data()
                rbm_prior = rbm_prior_critic[j]

                with tf.GradientTape() as tape:
                    x_fake = self.generator(rbm_prior)
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





            # RBM Training
            if self.train_rbm_every_n is None or epoch % self.train_rbm_every_n == 0: 
                for _ in range(self.update_ratios['rbm']):
                    real_data = self.sample_real_data()
                    if epoch < self.train_rbm_cutoff_epoch and epoch >= self.train_rbm_start_epoch:
                        # Generate Input for RBM Training
                        random_indices = np.random.choice(len(real_data), self.samples_train_rbm, replace=False)
                        feature_layer_output = self.feature_layer_model.predict(real_data[random_indices], verbose=0)
                        rbm_input = self.preprocess_rbm_input(feature_layer_output)

                        rbm_loss = self.rbm.train(rbm_input, num_reads=10)  

                    # Generate Training Data for Generator and Critic
                    rbm_prior = self.generate_prior(n_batches=self.update_ratios['critic'] + self.update_ratios['generator'])
                    rbm_prior_critic = rbm_prior[:self.update_ratios['critic']]
                    rbm_prior_generator = rbm_prior[self.update_ratios['critic']:]
                    
            rbm_loss_total += rbm_loss



            # Generator Training
            # for j in range(self.update_ratios['generator']):
            #     rbm_prior = rbm_prior_generator[j]
            #     g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

            # Generator Training
            for j in range(self.update_ratios['generator']):
                rbm_prior = rbm_prior_generator[j]

                with tf.GradientTape() as tape:
                    x_fake = self.generator(rbm_prior)
                    critic_x_fake = self.critic(x_fake)
                    mmd = self.compute_mmd(critic_x_real, critic_x_fake)
                    generator_loss = mmd
                generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                average_gen_loss += generator_loss




            # Save losses
            if epoch % self.save_frequency == 0:

                average_gen_loss /= self.save_frequency
                average_critic_loss /= self.save_frequency
                rbm_loss_total /= self.save_frequency
                self.generator_losses.append(average_gen_loss)
                self.critic_losses.append(average_critic_loss)
                self.rbm_losses.append(rbm_loss_total)
                print(f"Epoch {epoch}: generator MMD = {average_gen_loss:.4f}, critic MMD = {average_critic_loss:.4f}, RBM Loss: {rbm_loss_total:.4f}")
                average_gen_loss = 0
                average_critic_loss = 0
                rbm_loss_total = 0

    def train_2(self):
        real_data = self.sample_real_data()
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        d_loss_real_total = 0
        d_loss_fake_total = 0
        g_loss_total = 0

        # Train Simple GAN First
        for epoch in range(self.epochs):
            for _ in range(self.update_ratio_critic):
                # Train discriminator on real data
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                noise = self.sample_noise()
                generated_data = self.generator.predict(noise, verbose=0)
                fake_labels = np.zeros((self.batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # Train generator
            noise = self.sample_noise()
            g_loss = self.gan.train_on_batch(noise, real_labels)

            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            g_loss_total += g_loss

            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real_total}, D_loss_fake: {d_loss_fake_total}, G_loss: {g_loss_total}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0
        
        # Throw away the trained Generator 
        # self.generator = self.create_generator()
        # self.gan = self.create_gan()
        # self.compile_models()

        # Train the RBM which will serve as Prior
        print('Training the RBM')
        feature_layer_output = self.feature_layer_model.predict(real_data[:], verbose=0)
        rbm_input = self.preprocess_rbm_input(feature_layer_output)
        rbm_loss = self.rbm.train(rbm_input) 
        print('Final RBM loss:', rbm_loss) 

        # Train the Generator Alone against Discriminator
        # print('Training Generator Alone')
        # for _ in range(self.epochs):
        #     rbm_prior = self.generate_prior()
        #     g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

        # Train Complete GAN with RBM as Prior
        for epoch in range(self.epochs):
            for _ in range(self.update_ratio_critic):
                # Train discriminator on real data
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)

                # Train discriminator on generated data
                rbm_prior = self.generate_prior()
                generated_data = self.generator.predict(rbm_prior, verbose=0)
                fake_labels = np.zeros((self.batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)

            # Train generator
            rbm_prior = self.generate_prior()
            g_loss = self.gan.train_on_batch(rbm_prior, real_labels)

            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            g_loss_total += g_loss

            # Save losses
            if epoch % self.save_frequency == 0:
                d_loss_real_total /= self.save_frequency
                d_loss_fake_total /= self.save_frequency
                g_loss_total /= self.save_frequency
                print(f"Epoch {epoch}, D_loss_real: {d_loss_real_total}, D_loss_fake: {d_loss_fake_total}, G_loss: {g_loss_total}")
                self.d_losses_real.append(d_loss_real_total)
                self.d_losses_fake.append(d_loss_fake_total)
                self.g_losses.append(g_loss_total)

                d_loss_real_total = 0
                d_loss_fake_total = 0
                g_loss_total = 0


    def save_parameters_to_json(self, folder_path):
        parameters = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': self.hyperparameters,
        }
        with open(os.path.join(folder_path, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)

    def plot_losses(self, folder_path):
        plt.plot(self.critic_losses, label='Critic MMD')
        plt.plot(self.generator_losses, label='Generator MMD')
        plt.plot(self.rbm_losses, label='RBM Loss')
        plt.xlabel('Epoch x '+str(self.save_frequency))
        plt.ylabel('MMD')
        plt.legend()
        plt.title('MMDs')
        # plt.ylim(0, 2)

        plt.savefig(f'{folder_path}losses.png', dpi=300)
        plt.close()

    def plot_results(self, folder_path):
        rbm_prior = self.generate_prior(n_samples=1000)
        x_fake = self.generator(rbm_prior).numpy().flatten()

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




def simple_main(hyperparameters):
    gan = MMD_QAAAN(hyperparameters)
    gan.train()
    gan.plot_and_save()

def complex_main(hyperparameters):
    feature_layer_sizes = [1, 5, 10, 20]
    num_hiddens = feature_layer_sizes
    hyperparameters['hyperparameters_gan']['plotting']['results_path'] = 'results/2-tests/e_vanilla_qaaan/f_classical_different_rbm_sizes/'

    for i in feature_layer_sizes:
        hyperparameters['hyperparameters_qaaan']['feature_layer_size'] = i
        hyperparameters['hyperparameters_gan']['network']['latent_dim'] = i
        hyperparameters['hyperparameters_rbm']['network']['num_visible'] = i
        for j in num_hiddens:
            hyperparameters['hyperparameters_rbm']['network']['num_hidden'] = j 

            gan = ClassicalQAAAN(hyperparameters)
            gan.train()
            gan.plot_and_save()


def create_hyperparameters_mmd_gan(hyperparams_qaaan):
    return {
        'training': {
            'epochs': hyperparams_qaaan['training']['total_epochs'],
            'batch_size': hyperparams_qaaan['training']['batch_size'],
            'save_frequency': hyperparams_qaaan['training']['save_frequency'],
            'update_ratio_critic': hyperparams_qaaan['training']['update_ratios']['critic'],
            'update_ratio_gen': hyperparams_qaaan['training']['update_ratios']['generator'],
            'lr': hyperparams_qaaan['training']['mmd_gan_learning_rate'],
            'mmd_lamb': hyperparams_qaaan['training']['mmd_lamb'],
            'clip': hyperparams_qaaan['training']['clip'],
            'sigmas': hyperparams_qaaan['training']['sigmas']
        },
        'network': {
            'latent_dim': hyperparams_qaaan['network']['feature_layer_size'],
            'gen_hidden_units': hyperparams_qaaan['network']['layers_generator'],
            'critic_hidden_units': hyperparams_qaaan['network']['layers_critic'],
        },
        'distributions': {
            'mean': hyperparams_qaaan['distributions']['mean'],
            'variance': hyperparams_qaaan['distributions']['variance'],
            'target_dist': hyperparams_qaaan['distributions']['target_dist'],
            'input_dist': hyperparams_qaaan['distributions']['input_dist'],
        },
        'plotting': {
            'plot_size': hyperparams_qaaan['plotting']['plot_size'],
            'n_bins': hyperparams_qaaan['plotting']['n_bins'],
            'results_path': hyperparams_qaaan['plotting']['results_path'],
        },
    }

def create_hyperparameters_rbm(hyperparams_qaaan):
    return {
        'network': {
            'num_visible': hyperparams_qaaan['network']['feature_layer_size'],
            'num_hidden': hyperparams_qaaan['network']['rbm_num_hidden'],
            'qpu': True if hyperparams_qaaan['network']['rbm_type'] == 'quantum' else False,
        },
        'training': {
            'epochs': hyperparams_qaaan['training']['rbm_epochs'],
            'lr': hyperparams_qaaan['training']['rbm_learning_rate'],
            'verbose': hyperparams_qaaan['training']['rbm_verbose'],
        },
        'plotting': {
            'folder_path': hyperparams_qaaan['plotting']['rbm_folder_path'],
        }
    }


if __name__ == "__main__":
    one_run = True

    hyperparameters_qaaan = {
        'training': {
            'update_ratios': {
                'critic': 5,
                'generator': 1,
                'rbm': 1,
            },
            'total_epochs': 100,
            'train_rbm_every_n': 1,
            'train_rbm_cutoff_epoch': 50,
            'train_rbm_start_epoch': 1,
            'samples_train_rbm': 1,
            'batch_size': 100,
            'save_frequency': 1,
            'mmd_gan_learning_rate': 1e-3,
            'rbm_learning_rate': 0.01,
            'rbm_epochs': 1,
            'rbm_verbose': False,
            'mmd_lamb': 0.01,
            'clip': 1,
            'sigmas': [1, 2, 4, 8, 16],
        },
        'network': {
            'rbm_type': 'quantum',  # Can be classical, simulated, or quantum.
            'feature_layer_size': 20,
            'rbm_num_hidden': 20,
            'layers_generator': [2, 13, 7, 1],
            'layers_critic': [11, 29, 11],  
        },
        'plotting': {
            'plot_size': 10000,
            'n_bins': 100,
            'rbm_folder_path': None,
        },
        'distributions': {
            'mean': 1,
            'variance': 1,
            'target_dist': 'gaussian',
            'input_dist': 'uniform',
        },
    }

    hyperparameters_qaaan['plotting']['results_path'] = 'results/2-tests/f_mmd_qaaan/' + hyperparameters_qaaan['network']['rbm_type'] + '/'

    hyperparameters_mmd_gan = create_hyperparameters_mmd_gan(hyperparameters_qaaan)
    hyperparameters_rbm = create_hyperparameters_rbm(hyperparameters_qaaan)


    hyperparameters = {
        'hyperparameters_qaaan': hyperparameters_qaaan,
        'hyperparameters_mmd_gan': hyperparameters_mmd_gan,
        'hyperparameters_rbm': hyperparameters_rbm,
    }


    if one_run:
        simple_main(hyperparameters=hyperparameters)
    else:
        complex_main(hyperparameters=hyperparameters)


