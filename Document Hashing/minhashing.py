from nltk import ngrams
from MinHash import MinHash

def character_kgram(text, k):
	grams = ngrams(list(text), k)
	gram_set = set()
	for gram in grams:
		gram_set.add(''.join(gram))
	return gram_set

def build_hash_signature(grams, m):
	hash_sig = [None] * m
	for gram in grams:
		hash_sig[hash(gram) % m] = gram
	return hash_sig

t = [20, 60, 150, 300, 600]

D1 = open('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw2/D1.txt', 'r').read()
D2 = open('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw2/D2.txt', 'r').read()

d1 = character_kgram(D1, 3)
d2 = character_kgram(D2, 3)

for t_value in t:
	minhash_1 = MinHash(t_value)
	for gram in d1:
		minhash_1.add(gram)

	minhash_2 = MinHash(t_value)
	for gram in d2:
		minhash_2.add(gram)
	print 'JS_%d(D1, D2) = %.2f' % (t_value, minhash_1.jaccard(minhash_2))
