# Adapted from https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

from csv import reader
import os
import numpy as np
import pandas as pd

LEFT = 'LEFT'
RIGHT = 'RIGHT'

def load_data(file_name):
    instances = []
    for line in open(os.path.join(os.path.dirname(__file__), file_name)):
        tokens = line.split()
        label = tokens[0]
        line = line[2:]
        tokens = line.split(' ')
        features = np.zeros(17)

        for feature in tokens:
            feature_split = feature.split(':')
            index = int(feature_split[0]) - 1
            value = float(feature_split[1])
            features[index] = value

        features[-1] = label
        instances.append(features)

    return instances

def load_ids(file_name):
    ids = []
    for line in open(os.path.join(os.path.dirname(__file__), file_name)):
        ids.append(line)
    return ids


def score(actual, predicted):
	correct = 0
	for (y, prediction) in zip(actual, predicted):
		if y == prediction:
			correct = correct + 1
	return correct / float(len(actual)) * 100.0


def test_split(index, value, dataset):
	left = list()
	right = list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	n_instances = float(sum([len(group) for group in groups]))
	
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		
		gini += (1.0 - score) * (size / n_instances)
	return gini


def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	
	if not left or not right:
		node[LEFT] = node[RIGHT] = to_terminal(left + right)
		return
	
	if depth >= max_depth:
		node[LEFT], node[RIGHT] = to_terminal(left), to_terminal(right)
		return
	
	if len(left) <= min_size:
		node[LEFT] = to_terminal(left)
	else:
		node[LEFT] = get_split(left)
		split(node[LEFT], max_depth, min_size, depth + 1)
	
	if len(right) <= min_size:
		node[RIGHT] = to_terminal(right)
	else:
		node[RIGHT] = get_split(right)
		split(node[RIGHT], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root


def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node[LEFT], dict):
			return predict(node[LEFT], row)
		else:
			return node[LEFT]
	else:
		if isinstance(node[RIGHT], dict):
			return predict(node[RIGHT], row)
		else:
			return node[RIGHT]


def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	return([predict(tree, row) for row in test])


def main():
	train_instances = load_data('data-splits/data.train')
	test_instances = load_data('data-splits/data.test')
	train_instances += test_instances
	anon_instances = load_data('data-splits/data.eval.anon')
	anon_ids = load_ids('data-splits/data.eval.id')
	predictions = decision_tree(train_instances, anon_instances, 9, 8)

	rows = []
	for (anon_id, prediction) in zip(anon_ids, predictions):
	    rows.append([anon_id, prediction])

	df = pd.DataFrame(rows)
	df.to_csv("decision_tree_12_predictions.csv", header=['Id','Prediction'])

if __name__ == "__main__":
   
   main()
	
