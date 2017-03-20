# Implementation of hierarchical aggolmerative clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

c1_data = pd.read_csv('/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw3/C1.txt', sep="\t", header = None)

c1_data = c1_data.drop(0, axis=1)

# L2-norm distance between to points(arrays)
def l_two(a, b):
	sum = 0
	for v1, v2 in zip(a, b):
		sum += (v1 - v2)**2

	return sum

# Single-Link Distance Function 
def single_link(clusters):
	s1 = []
	s2 = []
	min_distance = sys.maxint

	for cluster in clusters:
		for point in cluster:
			for other_cluster in clusters:
				if other_cluster == cluster:
					continue
				for other_point in other_cluster:
					distance = l_two(point, other_point)
					if min_distance > distance:
						min_distance = distance
						s1 = cluster
						s2 = other_cluster
	return s1, s2

def complete_link(clusters):
	min_distance_tuple = sys.maxint, None, None

	for cluster in clusters:
		for point in cluster:
			max_point_dist = -sys.maxint, None, None
			for other_cluster in clusters:
				if other_cluster == cluster:
					continue
				for other_point in other_cluster:
					distance = l_two(point, other_point)
					if max_point_dist[0] < distance:
						max_point_dist = distance, cluster, other_cluster

				if min_distance_tuple[0] > max_point_dist[0]:
					min_distance_tuple = max_point_dist

	return min_distance_tuple[1], min_distance_tuple[2]

def mean_link(clusters):
	cluster_means = [[sum(y) / len(y) for y in zip(*cluster)] for cluster in clusters]
	s1 = ()
	s2 = ()
	min_distance = sys.maxint

	for point in cluster_means:
		for other_point in cluster_means:
			if other_point == point:
				continue
			distance = l_two(point, other_point)
			if min_distance > distance:
				min_distance = distance
				s1 = point
				s2 = other_point

	return clusters[cluster_means.index(s1)], clusters[cluster_means.index(s2)]

def hierarchical_clustering(data_frame, distance_func, k):
	clusters = [[tuple(x)] for x in data_frame.values]

	while len(clusters) > k:
		s1, s2 = distance_func(clusters)
		clusters.remove(s1)
		clusters.remove(s2)
		clusters.append(s1 + s2)
	return clusters

def plot_clusters(clusters, title):

	colors = ['r', 'g', 'b', 'orange']
	colorCount = 0
	for cluster in clusters:
		for point in cluster:
			plt.scatter(point[0], point[1], color=colors[colorCount])
		colorCount += 1

	plt.title(title)
	plt.savefig(title)


single_link_clusters = hierarchical_clustering(c1_data, single_link, 4)
plot_clusters(single_link_clusters, 'Hierarchical Clustering: Single-Link')

complete_link_clusters = hierarchical_clustering(c1_data, complete_link, 4)
plot_clusters(complete_link_clusters, 'Hierarchical Clustering: Complete-Link')

mean_link_clusters = hierarchical_clustering(c1_data, mean_link, 4)
plot_clusters(mean_link_clusters, 'Hierarchical Clustering: Mean-Link')
