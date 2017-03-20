# Implementation of lloyds algorithm for clustering.
import pandas as pd
import gonzalez as gz
import kmeans_plus_plus as kpp
import matplotlib.pyplot as plt
import numpy as np
import random

def lloyds(X, k, centers_function):
	centers = centers_function(X, k)
	prev_centers = []
	clusters = [[] for i in range(k)]

	while centers != prev_centers:
		clusters = [[] for i in range(k)]
		for x in X:
			closest_center, distance, index = gz.phi(x, centers)
			clusters[index].append(x)

		prev_centers = centers
		for i in range(0, k):
			centers[i] = np.mean(clusters[i])

	return centers, clusters

def random_centers(X, k):
	centers = []
	for i in range(0, k):
		centers.append(X[i])

	return centers

def plot_clusters(clusters, title):
	colors = ['r', 'g', 'b']
	colorIndex = 0

	for cluster in clusters:
		for point in cluster:
			plt.scatter(point[0], point[1], color=colors[colorIndex])
		colorIndex += 1

	plt.title(title)
	plt.savefig('%s.pdf' % title)

def main():
	c2_data = pd.read_csv('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw3/C2.txt', sep="\t", header=None)
	c2_data = c2_data.drop(0, axis=1)
	X = [tuple(x) for x in c2_data.values]

	k = 3

	centers, clusters = lloyds(X, k, gz.gonzalez_init)
	plot_clusters(clusters, 'Lloyd\'s with Gonzalez')

	centers, clusters = lloyds(X, k, random_centers)
	plot_clusters(clusters, 'Lloyd\'s with random centers')

	centers, clusters = lloyds(X, k, kpp.kmeans_plus_plus)
	plot_clusters(clusters, 'Lloyd\'s with k-Means++')

main()