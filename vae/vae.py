#!/usr/bin/python3
"""
Pytorch Variational Autoendoder Network Implementation
"""
from itertools import chain
import time
import json
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb


class Encoder(nn.Module):
    """
    Probabilistic Encoder

    Return the mean and the variance of z ~ q(z|x). The prior
    of x is assume to be normal(0, I).

    Arguments:
        input_dim {int} -- number of features

    Returns:
        (tensor, tensor) -- mean and variance of the latent variable
            output from the forward propagation
    """
    def __init__(self, input_dim, config):
        super(Encoder, self).__init__()

        config_encoder = json.loads(config.get("encoder"))
        config_read_mu = json.loads(config.get("read_mu"))
        config_read_logvar = json.loads(config.get("read_sigma"))

        config_encoder[0]['in_features'] = input_dim

        encoder_network = []
        for layer in config_encoder:
            if layer['type'] == 'linear':
                encoder_network.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'relu':
                encoder_network.append(nn.ReLU())
            elif layer['type'] == 'tanh':
                encoder_network.append(nn.Tanh())
            elif layer['type'] == 'dropout':
                encoder_network.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'batch_norm':
                encoder_network.append(nn.BatchNorm1d(layer['num_features']))

        self.encoder_network = nn.Sequential(*encoder_network)
        self.read_mu = nn.Linear(config_read_mu['in_features'], config.getint('latent_dim'))
        self.read_logvar = nn.Linear(config_read_logvar['in_features'], config.getint('latent_dim'))
        self.initialize_parameters()
        

    def initialize_parameters(self):
        """
        Xavier initialization
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        """
        Forward propagation
        """
        hidden_state = self.encoder_network(inputs)
        mean = self.read_mu(hidden_state)
        logvar = self.read_logvar(hidden_state)
        return mean, logvar


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, input_dim, config):
        super(Decoder, self).__init__()
        config_decoder = json.loads(config.get("decoder"))

        decoder_network = []
        for layer in config_decoder:
            if layer['type'] == 'linear':
                decoder_network.append(nn.Linear(layer['in_features'], layer['out_features']))
            elif layer['type'] == 'relu':
                decoder_network.append(nn.ReLU())
            elif layer['type'] == 'relu6':
                decoder_network.append(nn.ReLU6())
            elif layer['type'] == 'tanh':
                decoder_network.append(nn.Tanh())
            elif layer['type'] == 'sigmoid':
                decoder_network.append(nn.Sigmoid())
            elif layer['type'] == 'dropout':
                decoder_network.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'batch_norm':
                decoder_network.append(nn.BatchNorm1d(layer['num_features']))
            elif layer['type'] == 'read_x':
                decoder_network.append(nn.Linear(layer['in_features'], input_dim))
        self.decoder = nn.Sequential(*decoder_network)
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """
    def __init__(self, input_dim, config, checkpoint_directory):
        super(VAE, self).__init__()
        self.config = config
        self.model_name = '{}{}'.format(config['model']['name'], config['model']['config_id'])
        self.checkpoint_directory = checkpoint_directory
        self._device = config['model']['device']
        self._encoder = Encoder(input_dim, config['model'])
        self._decoder = Decoder(input_dim, config['model'])

        self.num_epochs = config.getint('training', 'n_epochs')

        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

        self.mu = None
        self.logvar = None

        self.precentile_threshold = config.getfloat('model', 'threshold')
        self.threshold = None

        self.cur_epoch = 0
        self._save_every = config.getint('model', 'save_every')


    def parameters(self):
        return chain(self._encoder.parameters(), self._decoder.parameters())

    def _sample_z(self, mu, logvar):
        epsilon = torch.randn(mu.size())
        epsilon = Variable(epsilon, requires_grad=False).type(torch.FloatTensor).to(self._device)
        sigma = torch.exp(logvar / 2)
        return mu + sigma * epsilon

    def forward(self, inputs):
        """
        Forward propagation
        """
        self.mu, self.logvar = self._encoder(inputs)
        latent = self._sample_z(self.mu, self.logvar)
        theta = self._decoder(latent)
        return theta

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1

            # temporary storage
            train_losses, train_kldivs, train_reconlos = [], [], []
            time_pre = time.time()
            batch = 0
            for inputs, _ in trainloader:
                self.train()
                inputs = inputs.to(self._device)
                outputs = self.forward(inputs)

                recon_loss = nn.L1Loss(reduction='sum')(outputs, inputs) / inputs.shape[0]
                kl_div = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) / inputs.shape[0]
                loss = recon_loss + kl_div
                loss.backward()

                self._optim.step()
                self._optim.zero_grad()

                train_losses.append(self._to_numpy(loss))
                train_kldivs.append(self._to_numpy(kl_div))
                train_reconlos.append(self._to_numpy(recon_loss))
                batch += 1
            
                if batch == 100:
                    print("Time: ", time.time() - time_pre)


            print('Training Loss: ', np.mean(train_losses))
            print('Train kldiv', np.mean(train_kldivs))
            print('Training Reconstruction', np.mean(train_reconlos))

            if self.config.getboolean("log", "wandb") is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Train kldiv'] = np.mean(train_kldivs)
                wandb_dict['Training Reconstruction'] = np.mean(train_reconlos)

                if epoch % 10 == 0:
                    f1, accuracy, precision, recall, recon_mean = self.evaluate(trainloader, validationloader)
                    self.save_checkpoint(f1)
                    print("Threshold: ", self.threshold)
                    print("Validation F1 Score: ", f1)
                    print("Validation Accuracy: ", accuracy)
                    print("Validation Precision: ", precision)
                    print("Validation Recall: ", recall)
                    print("Validation Recon Loss: ", recon_mean)
                    wandb_dict['Validation F1'] = f1
                    wandb_dict['Validation Accuracy'] = accuracy
                    wandb_dict['Validation Precision'] = precision
                    wandb_dict['Validation Recall'] = recall
                    wandb_dict['Validation Mean'] = recon_mean

                wandb.log(wandb_dict)

    def test(self, trainloader, testcollisionloader, testfreeloader):
        print("--------------------------------------------------------")
        print("Final Threshold: ", self.threshold)
        training_recon_losses = self._get_densities(trainloader)

        predictions_col = []
        ground_truth_col = []
        test_recon_losses_col = []
        test_recon_col = []
        for inputs, labels in testcollisionloader:
            pred, test_recon_loss, output = self.predict(inputs,test_mode=True)
            predictions_col.extend(pred)
            ground_truth_col.extend(list(self._to_numpy(torch.flatten(labels))))
            test_recon_losses_col.extend(test_recon_loss)
            test_recon_col.extend(output)
        
        f1 = f1_score(ground_truth_col, predictions_col)
        accuracy = accuracy_score(ground_truth_col, predictions_col)
        precision = precision_score(ground_truth_col, predictions_col)
        recall = recall_score(ground_truth_col, predictions_col)

        print("Test F1 Score: ", f1)
        print("Test Accuracy: ", accuracy)
        print("Test Precision: ", precision)
        print("Test Recall: ", recall)

        predictions_free = []
        test_recon_losses_free = []
        test_recon_free = []
        for inputs, labels in testfreeloader:
            pred, test_recon_loss, output = self.predict(inputs,test_mode=True)
            predictions_free.extend(pred)
            test_recon_losses_free.extend(test_recon_loss)
            test_recon_free.extend(output)

        np.savetxt("training_recon_loss.csv", training_recon_losses, delimiter=",")
        np.savetxt("test_collision_prediction.csv", predictions_col, delimiter=",")
        np.savetxt("test_collision_recon_loss.csv", test_recon_losses_col, delimiter=",")
        np.savetxt("test_collision_recon_result.csv", test_recon_col, delimiter=",")
        np.savetxt("test_free_prediction.csv", predictions_free, delimiter=",")
        np.savetxt("test_free_recon_loss.csv", test_recon_losses_free, delimiter=",")
        np.savetxt("test_free_recon_result.csv", test_recon_free, delimiter=",")
        


    def _get_densities(self, dataloader):
        all_recon_losses = []
        for inputs, _ in dataloader:
            mini_batch_recon_loss = self._evaluate_reconstruction_loss(inputs)
            all_recon_losses.extend(mini_batch_recon_loss)
        all_recon_losses = np.array(all_recon_losses)
        return all_recon_losses

    def _evaluate_reconstruction_loss(self, inputs, test_mode=False):
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(self._device)
            outputs = self.forward(inputs)
            recon_loss = nn.L1Loss(reduction='none')(outputs, inputs)
            recon_losses = torch.sum(recon_loss, 1)
            assert inputs.shape[0] == recon_losses.shape[0]
            if not test_mode:
                return self._to_numpy(recon_losses)
            else:
                return self._to_numpy(recon_losses), self._to_numpy(outputs)

    def _find_threshold(self, dataloader):
        densities = self._get_densities(dataloader)
        lowest_density = np.argmin(densities)
        self.threshold = np.percentile(densities, self.precentile_threshold)
        return lowest_density

    def evaluate(self, trainloader, validationloader):
        """
        Evaluate accuracy.
        """
        self._find_threshold(trainloader)
        predictions = []
        ground_truth = []
        recon_losses = []

        for inputs, labels in validationloader:
            pred, recon_loss = self.predict(inputs,test_mode=False)
            predictions.extend(pred)
            recon_losses.extend(recon_loss)
            ground_truth.extend(list(self._to_numpy(torch.flatten(labels))))

        f1 = f1_score(ground_truth, predictions)
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)
        recon_mean = np.mean(recon_losses,0)

        return f1, accuracy, precision, recall, recon_mean

    def predict(self, inputs, test_mode=False):
        """
        Predict the class of the inputs
        """
        if not test_mode:
            recon_loss = self._evaluate_reconstruction_loss(inputs,test_mode)
        else:
            recon_loss, outputs = self._evaluate_reconstruction_loss(inputs,test_mode)
        predictions = np.zeros_like(recon_loss).astype(int)
        predictions[recon_loss > self.threshold] = 1
        if not test_mode:
            return list(predictions), recon_loss
        else:
            return list(predictions), recon_loss, outputs

    def save_checkpoint(self, f1_score):
        """Save model paramers under config['model_path']"""
        model_path = '{}/epoch_{}-f1_{}.pt'.format(
            self.checkpoint_directory,
            self.cur_epoch,
            f1_score)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, epoch):
        """
        Retore the model parameters
        """
        model_path = '{}{}_{}.pt'.format(
            self.config['paths']['checkpoints_directory'],
            self.model_name,
            epoch)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_epoch = epoch
