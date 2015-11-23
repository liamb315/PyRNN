from random import *

rand_binary_fun = lambda n: [randint(0,1) for b in range(1,n+1)]
binary = rand_binary_fun(10000)


w1   = 'deep'
w2   = 'learning'
text = ''

for i in binary:
	if i == 0:
		text = text + w1 + ' '
	elif i == 1:
		text = text + w2 + ' '

f = open('../data/simple.txt', 'w')
f.write(text)
f.close()