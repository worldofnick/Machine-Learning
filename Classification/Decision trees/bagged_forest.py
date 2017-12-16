import math
import os
import numpy as np
from dt import ID3
from random import randrange

def setToN(n):
	numbers = set()
	for i in range(0, n):
		numbers.add(i)
	return numbers

def convert_collection(X, Y, feature_count):
    X_c = []
    Y_c = []
    for (x, y) in zip(X, Y):
        x_c, y_c = convert(x, y, feature_count)
        X_c.append(x_c)
        Y_c.append(y_c)

    return X_c, Y_c

def convert(x, y, feature_count):
    arr = np.zeros(feature_count)
    if type(x) == np.ndarray:
        return x, y
    for key, value in x.iteritems():
        arr[key - 1] = value

    return arr, y

def majorityElement(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def predict(roots, X_test):
    predictions = []
    for x in X_test:
        many_predictions = []
        for root in roots:
            many_predictions.append(root.predict(x))

        predictions.append(majorityElement(many_predictions))

    return predictions

def score(predictions, Y, title, depth):
    correct = 0
    for (p, y) in zip(predictions, Y):
        if type(p) == tuple:
            p = p[0]
        if p == 0:
            p = -1
        if p == y:
            correct += 1

    # print 'Bagged tree accuracy on %s with depth %d = %.3f' % (title, depth, correct/float(len(Y)))
    return correct/float(len(Y))

def random_data(X, Y, count=100):
    X_rand = []
    Y_rand = []

    for i in range(0, count):
        random_index = randrange(0, len(X))
        X_rand.append(X[random_index])
        Y_rand.append(Y[random_index])

    return X_rand, Y_rand

def merged_excluding(X_trains, Y_trains, index):
    X = []
    Y = []

    for i in range(0, len(X_trains)):
        if i != index:
            X = X + X_trains[i]
            Y = Y + Y_trains[i]

    return X, Y

def build_roots(X, Y, attrs, depth):
    roots = []
    N = 1000
    for i in range(0, N):
        X_rand, Y_rand = random_data(X, Y)
        root = ID3(X_rand, Y_rand, attrs, 0, maxDepth=depth, clear=True)
        roots.append(root)

    return roots

def run_bagged_forest(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
	print '************* REGULAR BAGGED FOREST *************'
	feature_count = 0
	for example in X_train:
	    feature_count = max(feature_count, max(example))

	for example in X_test:
	    feature_count = max(feature_count, max(example))

	X_train, Y_train = convert_collection(X_train, Y_train, feature_count)
	attrs = setToN(feature_count)

	for i in range(0, len(XCV)):
	    new_x, new_y = convert_collection(XCV[i], YCV[i], feature_count)
	    XCV[i] = new_x
	    YCV[i] = new_y


	hyperparams = [3]
	best_depth = 3
	best_avg = 0.0
	depth = best_depth

	roots = build_roots(X_train, Y_train, attrs, depth)
	predictions = predict(roots, X_train)
	accuracy = score(predictions, Y_train, 'train', depth)
	print 'Bagged forest train accuracy = %.3f' % accuracy

	X_test, Y_test = convert_collection(X_test, Y_test, feature_count)
	predictions = predict(roots, X_test)
	accuracy = score(predictions, Y_test, 'test', depth)
	print 'Bagged forest test accuracy = %.3f' % accuracy
	print '************* END BAGGED FOREST *************'
