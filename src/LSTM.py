import numpy as np


class LSTM:

    def __init__(self, char_to_inds, inds_to_char, vocab_size, hidden=100, seq_len=25,
                 epochs=1000, alpha=0.01, beta1=0.9, beta2=0.999):
        self.char_to_inds = char_to_inds  # mapping all characters to an index
        self.inds_to_char = inds_to_char  # mapping all the indices to a character
        self.vocab_size = vocab_size  # size unique elements in the text corpus (essentially vocabulary size)
        self.hidden = hidden  # size of the hidden layer
        self.seq_len = seq_len  # size of input sequence length (equivalent to batch size)
        self.epochs = epochs  # number of epochs to train for
        self.alpha = alpha  # learning rate
        self.beta1 = beta1  # 1st momentum parameter
        self.beta2 = beta2  # 2nd momentum parameter
        self.smooth_loss = -1 * np.log(1.0 / self.vocab_size) * self.seq_len
        self.params = {}  # initializing dictionary to hold weights and biases between all layers in a cell
        self.grads = {}  # initializing a dictionary to hold gradient values of wrights and biases during Backprop
        self.adam_params = {}  # initializing a dictionary to hold optimizer parameters
        self.init_weights_biases()
        self.init_optimizer()

    def init_weights_biases(self):
        std = (1.0 / np.sqrt(self.vocab_size + self.hidden))  # std for Xavier Initialization

        # Forget gate
        # weight matrix between input layer and hidden layer (100, 177)
        # input = previous hidden + current input = 100 + 77 = 177
        # since hidden units are constant as 100 and vocabulary size is 77
        self.params["Wf"] = np.random.randn(self.hidden, self.hidden + self.vocab_size) * std
        # bias added to all hidden units hence is vector of len hidden units
        self.params["bf"] = np.ones((self.hidden, 1))

        # Input gate
        # weight matrix between input layer and hidden layer (100, 177)
        # input = previous hidden + current input = 100 + 77 = 177
        # since hidden units are constant as 100 and vocabulary size is 77
        self.params["Wi"] = np.random.randn(self.hidden, self.hidden + self.vocab_size) * std
        # bias added to all hidden units hence is vector of len hidden units
        self.params["bi"] = np.ones((self.hidden, 1))

        # Cell gate
        # weight matrix between input layer and hidden layer (100, 177)
        # input = previous hidden + current input = 100 + 77 = 177
        # since hidden units are constant as 100 and vocabulary size is 77
        self.params["Wc"] = np.random.randn(self.hidden, self.hidden + self.vocab_size) * std
        # bias added to all hidden units hence is vector of len hidden units
        self.params["bc"] = np.ones((self.hidden, 1))

        # Output gate
        # weight matrix between input layer and hidden layer (100, 177)
        # input = previous hidden + current input = 100 + 77 = 177
        # since hidden units are constant as 100 and vocabulary size is 77
        self.params["Wo"] = np.random.randn(self.hidden, self.hidden + self.vocab_size) * std
        # bias added to all hidden units hence is vector of len of hidden units
        self.params["bo"] = np.ones((self.hidden, 1))

        # output
        # weight matrix between hidden and output layer (77, 100)
        # since hidden units are constant 100 and output vocabulary size is 77
        self.params["Wv"] = np.random.randn(self.vocab_size, self.hidden) * (1.0 / np.sqrt(self.hidden))
        # bias added to all the output units hence vector is len of vocabulary size
        self.params["bv"] = np.ones((self.vocab_size, 1))

    def init_optimizer(self):

        # loop over all the keys in the parameters dictionary and create a zeros matrix of the same shape
        # this is required so that during backpropagation, these dictionaries will hold optimizer gradients
        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.adam_params["m" + key] = np.zeros_like(self.params[key])
            self.adam_params["v" + key] = np.zeros_like(self.params[key])

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-1 * x))

    def softmax(self, x):
        """
        This is used to give the probability distribution
        :param x:
        :return:
        """
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    def clip_grads(self):
        """
        to prevent exploding gradients we clip the gradients to a particular value
        :return:
        """
        for key in self.grads:
            np.clip(self.grads[key], -5, 5, out=self.grads[key])

        return
