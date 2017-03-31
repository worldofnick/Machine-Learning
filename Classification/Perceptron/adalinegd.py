import numpy as np

class AdalineGD(object):
	""" 
	Parameters
	----------
	eta : float
		The learning rate 0 <= 1
	n_iter : int
		Passes over the training dataset

	Attributes
	----------
	w_ : 1d-array
		Weights after fitting
	errors_ : list
		Number of misclassifications in every epoc
	"""

	def __init__(self, eta=0.01, n_iter=50):
		self.eta = eta
		self.n_iter = n_iter


	def fit(self, X, y):
		""" Train the model on training data

		Parameters
		----------
		x : {x-dimensional array} shape -> [n_samaples, n_features]
			Training vectors

		y : array-like shape -> [n_samples]
			Target values for the traning vectors

		Returns
		-------
		self : object

		"""

		self.w_ = np.zeros(1 + X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
