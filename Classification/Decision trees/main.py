# Implementation of a Decision tree and the ID3 tree building algorithm

import math
import os
from features import featureize
from Node import Node, Leaf

# Entropy
def entropy(propPlus, propMinus):
	if propPlus == 0 or propMinus == 0:
		return 0
	entropy = -1 * (propPlus * math.log(propPlus,2)  + propMinus * math.log(propMinus,2)) 
	return entropy

# Majority error	
def majorityError(propPlus, pminus):
	majority_error = 1 - max(propPlus, pminus)
	return majority_error


def informationGain(X, Y, entrpy, feature_index):
	propPlusMinus, propMinusMinus = proportionLabels(X, Y, feature_index, 0)
	propPlusPlus, propMinusPlus = proportionLabels(X, Y, feature_index, 1)

	entropyForMinusAttribute = entropy(propPlusMinus, propMinusMinus)
	entropyForPlusAttribute = entropy(propPlusPlus, propMinusPlus)

	trueCount, falseCount = proportionFeatures(X, feature_index)
	size = trueCount + falseCount

	return entrpy - ((trueCount / float(size)) * entropyForPlusAttribute + (falseCount / float(size)) * entropyForMinusAttribute)


def proportionLabels(X, Y, feature_index=None, feature_value=None):
	plusCount = 0
	minusCount = 0
	size = 0

	for x, y in zip(X, Y):
		if y == '+' and (feature_index == None or x[feature_index] == feature_value):
			plusCount += 1
			size += 1
		elif y == '-' and (feature_index == None or x[feature_index] == feature_value):
			minusCount += 1
			size += 1

	if size == 0:
		return 0, 0
	return plusCount / float(size), minusCount / float(size)


def proportionFeatures(X, feature_index): 
	trueCount = 0
	falseCount = 0

	for x in X:
		if x[feature_index] == 0:
			falseCount += 1
		elif x[feature_index] == 1:
			trueCount += 1

	return trueCount, falseCount


def findBestFeatureIndex(X, Y, attrs):
	propPlus, propMinus = proportionLabels(X, Y)
	entrpy = entropy(propPlus, propMinus)
	maxIndex = -1
	maxInformationGain = -1

	for index in attrs:
		information_gain = informationGain(X, Y, entrpy, index)
		if information_gain > maxInformationGain:
			maxInformationGain = information_gain
			maxIndex = index

	return maxIndex


# Read in our data
def loadData(file_name):
	# Array of names
	X = []

	# Array of labels (+, -)
	Y = []

	for line in open(os.path.join(os.path.dirname(__file__), 'updated_test/' + file_name)):
		label = line[0]
		full_name = line[1: len(line) - 2].strip()
		X.append(featureize(full_name))
		Y.append(label)

	return X, Y


def splitData(X, Y, feature_index, feature_value):

	newX = []
	newY = []

	for x, y in zip(X, Y):
		if x[feature_index] == feature_value:
			newX.append(x)
			newY.append(y)

	return newX, newY

def majorityElement(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def ID3(X, Y, attrs, currentDepth, maxDepth=7):

	currentDepth += 1
	# If all examples are positive, Return the single-node tree Root, with label = +.
 	# If all examples are negative, Return the single-node tree Root, with label = -.
	if len(set(Y)) == 1:
		return Leaf(Y[0])

	# If the number of predicting attributes is empty, then return a single node tree with
	# Label = most common value of the target attribute in the examples
	if len(attrs) == 0 or currentDepth == maxDepth:
		return Leaf(majorityElement(Y))

	# A = The Attribute that best classifies examples.
	A = findBestFeatureIndex(X, Y, attrs)
	newAttrs = [attr for attr in attrs if attr != A]
	# Decision Tree attribute for Root = A.
	root = Node(A)

	# For each possible value, vi, of A
	for possible_value in [0, 1]:
		# Let Examples(vi) be the subset of examples that have the value vi for A
		X_s, Y_s = splitData(X, Y, A, possible_value)

		if len(X_s) == 0 or len(Y_s) == 0: 
			# Then below this new branch add a leaf node with label = most common target value in the examples
			root.addChild(possible_value, Leaf(majorityElement(Y)))
		else:
			# Else below this node add a subtree ID3(Examples_vi, TargetAttr, Attrs - {A})
			root.addChild(possible_value, ID3(X_s, Y_s, newAttrs, currentDepth, maxDepth))

	return root

def setToN(n):
	numbers = set()
	for i in range(0, n):
		numbers.add(i)
	return numbers

def score(root, X, Y):
	correct = 0
	for x, y in zip(X, Y):
		if root.predict(x) == y:
			correct += 1

	return correct/ float(len(Y))

def mergeExcluding(X_trains, Y_trains, index):
	X = []
	Y = []

	for i in range(0, len(X_trains)):
		if i != index:
			X = X + X_trains[i]
			Y = Y + Y_trains[i]

	return X, Y

X, Y = loadData('updated_train.txt')
attrs = setToN(len(X[0]))
root = ID3(X, Y, attrs, 0, maxDepth=4)


print 'Score on training data = %.3f' % score(root, X, Y)

X, Y = loadData('updated_test.txt')
print 'Score on testing data = %.3f' % score(root, X, Y)

print 'Depth of tree = %d\n\n' % root.depth()

# Cross-validation

# Load all our datasets
# X_test, Y_test = loadData('updated_test.txt')
X_train_1, Y_train_1 = loadData('Updated_CVSplits/updated_training00.txt')
X_train_2, Y_train_2 = loadData('Updated_CVSplits/updated_training01.txt')
X_train_3, Y_train_3 = loadData('Updated_CVSplits/updated_training02.txt')
X_train_4, Y_train_4 = loadData('Updated_CVSplits/updated_training03.txt')

X_trains = [X_train_1, X_train_2, X_train_3, X_train_3]
Y_trains = [Y_train_1, Y_train_2, Y_train_3, Y_train_3]

# k-Fold cross validation
k = 4

depths = [1, 2, 3, 4, 5, 10, 15, 20]

for max_depth in depths:
	for i in range(0, len(X_trains)):
		X_train = X_trains[i]
		Y_train = Y_trains[i]

		attrs = setToN(len(X_train[0]))
		root = ID3(X_train, Y_train, attrs, 0, maxDepth=max_depth)
		X_test, Y_test = mergeExcluding(X_trains, Y_trains, i)
		accuracy = score(root, X_test, Y_test)
		std = standardDeviation(root, X_test, Y_test)
		print 'Training on updated_training0%d with a max depth of %d resulted in %.3f accuracy.' % (i, root.depth(), accuracy)
		print 'Training on updated_training0%d with a max depth of %d resulted in %.3f standard deviation.' % (i, root.depth(), accuracy)


X, Y = loadData('updated_train.txt')
attrs = setToN(len(X[0]))
root = ID3(X, Y, attrs, 0, maxDepth=4)

print '\n'
X, Y = loadData('updated_test.txt')
print 'Score on testing data when max depth is 4 = %.3f' % score(root, X, Y)