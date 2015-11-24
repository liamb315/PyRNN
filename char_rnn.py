import numpy as np
import theano
import theano.tensor as T
from lstm import InputLayer, SoftmaxLayer, LSTMLayer
from lib import make_caches, get_params, SGD, momentum, one_step_updates


class CharRNN:
    def __init__(self):
        X = T.matrix()
        Y = T.matrix()
        eta = T.scalar()
        temperature=T.scalar()

        self.num_input = 256
        self.num_hidden = 500
        self.num_output = 256

        inputs = InputLayer(X, name="inputs")
        lstm1 = LSTMLayer(self.num_input, self.num_hidden, input_layer=inputs, name="lstm1")
        lstm2 = LSTMLayer(self.num_hidden, self.num_hidden, input_layer=lstm1, name="lstm2")
        softmax = SoftmaxLayer(self.num_hidden, self.num_output, input_layer=lstm2, name="yhat", temperature=temperature)

        Y_hat = softmax.output()

        self.layers = inputs, lstm1, lstm2, softmax

        params = get_params(self.layers)
        caches = make_caches(params)


        cost = T.mean(T.nnet.categorical_crossentropy(Y_hat, Y))
        updates = momentum(cost, params, caches, eta)

        self.train = theano.function([X, Y, eta, temperature], cost, updates=updates, allow_input_downcast=True)

        predict_updates = one_step_updates(self.layers)
        self.predict_char = theano.function([X, temperature], Y_hat, updates=predict_updates, allow_input_downcast=True)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
