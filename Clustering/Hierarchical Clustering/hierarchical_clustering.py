# Implementation of hierarchical aggolmerative clustering
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys

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
	s1 = []
	s2 = []
	max_distance = -sys.maxint

	for cluster in clusters:
		for point in cluster:
			for other_cluster in clusters:
				if other_cluster == cluster:
					continue
				for other_point in other_cluster:
					distance = l_two(point, other_point)
					if max_distance < distance:
						max_distance = distance
						s1 = cluster
						s2 = other_cluster
	return s1, s2

def hierarchical_clustering(data_frame, distance_func, k):
	clusters = [[tuple(x)] for x in data_frame.values]

	while len(clusters) > k:
		s1, s2 = distance_func(clusters)
		clusters.remove(s1)
		clusters.remove(s2)
		clusters.append(s1 + s2)
	return clusters

print hierarchical_clustering(c1_data, complete_link, 2)

