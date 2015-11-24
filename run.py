from char_rnn import CharRNN
from lib import one_hot, one_hot_to_string, floatX
import numpy as np
import theano
import theano.tensor as T
import sys
import random


#  Prepare the data
# f = open("../data/reuters21578/reut2-002.sgm")
#f = open("../data/warpeace_input.txt")
f = open("../data/simple.txt")
text = f.read()
f.close()

rnn = CharRNN()

seq_len = 150

def train(eta, iters):
    for it in xrange(iters):
        i = random.randint(0, len(text)/seq_len)
        j = i * seq_len
        X = text[j:(j+seq_len)]
        Y = text[(j+1):(j+1+seq_len)]
        print "iteration: %s, cost: %s" % (str(it), str(rnn.train(one_hot(X, rnn.num_input), one_hot(Y, rnn.num_output), eta, 1.0)))


def infer_stochastic(rnn, k, temperature, start_char=" "):
    x = [one_hot(start_char).flatten()]
    for i in xrange(k):
        probs = rnn.predict_char(x, temperature)
        p = np.asarray(probs[0], dtype="float64")
        p /= p.sum()
        sample = np.random.multinomial(1, p)
        sys.stdout.write(one_hot_to_string(sample))
        x = [sample]
    rnn.reset_state()
    print


if __name__=='__main__':
    print 'Training...'
    train(10, 25)
    print 'Generate stochastic text:'
    infer_stochastic(rnn, 10, 0.3, start_char='')