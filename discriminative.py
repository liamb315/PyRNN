import numpy as np
import theano
import theano.tensor as T
from lstm import InputLayer, SoftmaxLayer, LSTMLayer
from lib import make_caches, get_params, SGD, momentum, one_step_updates

class Discrimator:
	def __init__(self):
		X   = T.matrix()
		Y   = T.scalar()
		eta = T.scalar()

		self.num_input  = 256
		self.num_hidden = 500
		self.num_output = 1

		inputs  = InputLayer(X, name="inputs")
        lstm1   = LSTMLayer(self.num_input, self.num_hidden, input_layer=inputs, name="lstm1")
        lstm2   = LSTMLayer(self.num_hidden, self.num_hidden, input_layer=lstm1, name="lstm2")
        sigmoid = SigmoidLayer(input_layer=lstm2, name="yhat")

        Y_hat   = sigmoid.output()

        self.layers = inputs, lstm1, lstm2, sigmoid

        params      = get_params(self.layers)
        caches      = make_caches(params)

        # Target replication across the entire sequence to begin
        cost = T.mean