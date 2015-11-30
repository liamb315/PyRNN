from discriminative import Discriminator
from lib import one_hot, convert_numpy, one_hot_to_string, floatX
import numpy as np
import theano
import theano.tensor as T
import sys
import random

# Prepare the data
x_txt = open('../data/X_beer_reviews.txt')
text = x_txt.read()
x_txt.close()

y_txt = open('../data/Y_beer_reviews.txt')
labels = y_txt.read()
y_txt.close()

rnn = Discriminator()

seq_len = 150

def train(eta, iters):
	for it in xrange(iters):
		i = random.randint(0, (len(text)/seq_len -1)) # Ensure we don't exceed training corpus
		j = i * seq_len
		X = text[j:(j+seq_len)]
		Y = labels[j:(j+seq_len)]
		print "iteration: %s, cost: %s" % (str(it), str(rnn.train(one_hot(X, rnn.num_input), convert_numpy(Y), eta)))


if __name__ == '__main__':
	train(1, 10)