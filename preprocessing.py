X            = open('../data/X_beer_reviews.txt', 'w')
Y            = open('../data/Y_beer_reviews.txt', 'w')

with open('../data/fake_beer_reviews.txt', 'r') as fake_reviews:
	while True:
		c = fake_reviews.read(1)
		if not c:
			break
		X.write(c)
		Y.write('0')

with open('../data/real_beer_reviews.txt', 'r') as real_reviews:
	while True:
		c = real_reviews.read(1)
		if not c:
			break
		X.write(c)
		Y.write('1')


fake_reviews.close()
real_reviews.close()
X.close()
Y.close()