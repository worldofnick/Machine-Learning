# Adapated from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import os
import math
import operator
import pandas as pd
import numpy as np

from decisiontree import load_data, load_ids

def get_neighbors(traning_set, instance, k, distance_metric):
	distances = []

	for x in range(len(traning_set)):
		distances.append((traning_set[x], distance_metric(instance, traning_set[x])))

	distances.sort(key=operator.itemgetter(1))

	return [distances[x][0] for x in range(k)]

def get_vote(neighbors):
	class_votes = {}
	for x in range(len(neighbors)):
		vote = neighbors[x][-1]
		if vote in class_votes:
			class_votes[vote] += 1
		else:
			class_votes[vote] = 1
	return sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

def euclidean_distance(instance_1, instance_2):
    distance = 0
    for i in range(len(instance_1) - 1):
        distance += pow((instance_1[i] - instance_2[i]), 2)
    return math.sqrt(distance)


def manhattan_distance(instance_1, instance_2):
    distance = 0
    for i in range(len(instance_1) - 1):
        distance += abs(instance_1[i] - instance_2[i])
    return distance


def chebyshev_distance(instance_1, instance_2):
    distance = 0
    for i in range(len(instance_1) - 1):
        distance = max(distance, abs(instance_1[i] - instance_2[i]))
    return distance


train_instances = load_data('data-splits/data.train')
test_instances = load_data('data-splits/data.test')
train_instances += test_instances
anon_instances = load_data('data-splits/data.eval.anon')
anon_ids = load_ids('data-splits/data.eval.id')

rows = []

for (test_id, test_instance) in zip(anon_ids, anon_instances):
    neighbors = get_neighbors(train_instances, test_instance, 3, euclidean_distance)
    rows.append([test_id, get_vote(neighbors)])

df = pd.DataFrame(rows)
df.to_csv("knn_super_predictions.csv", header=['Id','Prediction'], index=False)
