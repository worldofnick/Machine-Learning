import numpy as np

class Perceptron(object):
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

	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, x, y):
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

		self._w = np.zeros(1 + x.shape[1]) # A weight for each feature
		self._errors = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(x,y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self._errors.append(errors)
		return self

	def net_input(self, x):
		""" Calculate net input """
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, x):
		""" Return class label after unit step """
		return np.where(self.net_input(x) >= 0.0, 1, -1)

			