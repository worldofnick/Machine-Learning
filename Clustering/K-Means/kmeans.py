# Nick Porter, CS 4964 - Math for Data HW 5 Q1, University of Utah
# k-means clustering via lloyds algorithm

import pandas as pandas
import numpy as np
import random
from numpy import linalg as LA
from sklearn.decomposition import PCA

# X = data, k = number of sites
def init_sites(X, k):
	random_data = random.sample(range(0, len(X)), k)
	sites = []

	for i in range(0, len(random_data)):
		sites.append((X[random_data[i]].flatten().tolist()))
	return sites

# Controls when the clustering loop will stop
def converged(sites, old_sites, iterations):
	if iterations > 1000:
		return True
	return old_sites == sites

# Maps each data point to the closest site
def map_to_site(X, sites):

	clusters = [[] for i in range(len(sites))]

	# For each data point find which site is the closest
	for i in range(len(X)):
		data_point = X[i]
	
	return clusters

# K-means clustering
# X = data, k = number of clusters
def kmeans(X, k):
	# choose k points S in X
	sites = init_sites(X, k)
	old_sites = [[] for i in range(0, k)]
	iterations = 0

	while not converged(sites, old_sites, iterations):
		iterations += 1

		# assign each point to the closest site
		clusters = map_to_site(X, sites)
		for i in range(0, k):
			old_sites[i] = sites[i]
			sites[i] = np.mean(clusters[i], axis=0).tolist()
	return sites

P = pandas.read_csv('P.csv')
Q = pandas.read_csv('Q.csv')
P = P.as_matrix()
Q = Q.as_matrix()

kmeans = kmeans(P, 3)