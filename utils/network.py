"""Class that represents the network to be evolved."""
import random
import logging
from utils.train import train_and_score


class Network:
    """
    Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """
        Initialize our network
        :param nn_param_choices: Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = dict()  # represents MLP network parameters
        self.id = ''  # id use to load the model later on
        self.model = ''  # compiled model

    def create_random(self):
        """
        Create a random network
        """
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """
        Set network properties
        :param network: The network parameters
        :return: None.
        """
        self.network = network

    def train(self, x_train, x_test, y_train, y_test):
        """
        Train the network and record the accuracy
        """
        if self.accuracy == 0.:
            self.id, self.model, self.accuracy = train_and_score(self.network, x_train, x_test, y_train, y_test)

    def print_network(self):
        """
        Print out a network
        """
        msg = 'model id = {0}'.format(self.id)
        logging.getLogger('regular').info(msg)
        logging.getLogger('regular').info(self.network)
        msg = 'Network accuracy = {0:.2f}'.format(self.accuracy * 100)
        logging.getLogger('regular').info(msg)
