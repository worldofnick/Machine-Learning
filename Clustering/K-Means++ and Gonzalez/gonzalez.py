# Implementation of the gonzalez algorithm to solve the k-centers problem
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

def three_center_cost(X, centers):
	max_dist = -sys.maxsize
	point = None
	center = None
	for x in X:
		cluster_center, distance = phi(x, centers)
		if distance > max_dist:
			max_dist = distance
			point = x
			center = centers[cluster_center]
	return max_dist, point, center

def three_means_cost(X, centers):
	sum = 0

	for x in X:
		cluster_center, distance = phi(x, centers)
		sum += distance**2

	average = sum / len(X)
	return average**0.5

# Returns the closest cluster center to x along with the distance
def phi(x, clusters):
	argmin, distances = pairwise_distances_argmin_min([x], clusters, metric='euclidean')
	return argmin[0], distances[0], argmin

def gonzalez_init(X, k):
	clusters = [X[0]]
	
	for i in range(1, k):
		max_dist = -sys.maxsize
		next_center = None
		for x in X:
			cluster_center, distance, index = phi(x, clusters)
			if distance > max_dist:
				max_dist = distance
				next_center = x
		clusters.append(next_center)

	return clusters

def plot_clusters(points, centers, title):

	for point in points:
		plt.scatter(point[0], point[1], alpha=0.3)
	for center in centers:
		plt.scatter(center[0], center[1], color='r', s=50)

	plt.title(title)
	plt.show()

def main():
	c2_data = pd.read_csv('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw3/C2.txt', sep="\t", header = None)
	c2_data = c2_data.drop(0, axis=1)
	X = [tuple(x) for x in c2_data.values]
	centers = gonzalez_init(X, 3)

	print 'Centers after running Gonzalez %s' % centers
	three_center_dist, point, center = three_center_cost(X, centers)
	print 'Three Center Cost: %s d(%s, %s)' % (three_center_dist, point, center)

	three_means_dist = three_means_cost(X, centers)
	print 'Three Means Cost: %s ' % (three_means_dist)
	plot_clusters(X, centers, 'Gonzalez for k-centers')


#main()
