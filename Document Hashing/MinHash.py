import numpy as np

class MinHash(object):
	def __init__(self, k, seed=10):
		self.k = k
		self.seed = seed

		minint = np.iinfo(np.int64).min
		maxint = np.iinfo(np.int64).max

		self.masks = (np.random.RandomState(seed=self.seed).randint(minint, maxint, k))

		self.hashes = np.empty(self.k, dtype=np.int64)
		self.hashes.fill(maxint)


	def add(self, v):
		hashes = np.bitwise_xor(self.masks, hash(v))
		self.hashes = np.minimum(self.hashes, hashes)

	def jaccard(self, other):
		if np.any(self.masks != other.masks):
			print 'Can only calculate similarity bewteen Minhashes with same hash functions'

		return (self.hashes == other.hashes).sum() / float(self.k)