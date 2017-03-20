# Implementation of the k-means++ algorithm to solve the k-centers problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gonzalez import phi
import gonzalez as gz
import random

def pick_with_probablity(X, clusters):

	distances = []
	sum = 0

	for x in X:
		cluster, squared_distance, index = gz.phi(x, clusters)
		squared_distance = squared_distance**2
		distances.append(squared_distance)
		sum += squared_distance

	distances = np.array(distances) / float(sum)
	cumsum = np.cumsum(distances)

	rand = random.uniform(0, 1)
	for i in range(0, len(cumsum)):
		if cumsum[i] > rand:
			return X[i]



def kmeans_plus_plus(X, k):
	clusters = [X[0]]

	for i in range(1, k):
		clusters.append(pick_with_probablity(X, clusters))

	return clusters

def main():
	c2_data = pd.read_csv('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw3/C2.txt', sep="\t", header = None)
	c2_data = c2_data.drop(0, axis=1)
	X = [tuple(x) for x in c2_data.values]
	
	match_gz = 0
	gz_centers = gz.gonzalez_init(X, 3)
	iterations = 50
	
	costs = []

	for i in range(0, iterations):
		centers = kmeans_plus_plus(X, 3)
		costs.append(gz.three_means_cost(X, centers))
	
	# Plot
	x_values = np.sort(costs)
	y_values = []
	y_value = 0
	for i in range(0, len(costs)):
		y_value += 1/ float(len(x_values))
		y_values.append(y_value)
	
	plt.plot(x_values, y_values)
	plt.xlabel("3-means cost")
	plt.ylabel("Percent of data <= to the corresponding 3-means cost")
	plt.title("CDF plot of k-Means++ 3-means cost ")
	plt.show()	

	
	print centers	

#main()
