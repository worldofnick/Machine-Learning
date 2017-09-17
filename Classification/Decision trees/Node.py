class Node:

	def __init__(self, attribute):
		self.attribute = attribute
		self.children = {}

	def addChild(self, value, node):
		self.children[value] = node

	# Chooses a label based on attribute of this node
	def pick(self, attribute_example):
		return self.children[attribute_example[self.attribute]]

	# The depth of the tree starting at this node
	def depth(self):
		maxValue = 0
		if 0 in self.children:
			maxValue = max(self.children[0].depth(), maxValue)

		if 1 in self.children:
			maxValue = max(self.children[1].depth(), maxValue)
		return maxValue + 1

	# Returns True if all children are leafs, false otherwise
	def areChildrenLeafs(self):
		for child in self.children.values():
			if not child.isLeaf():
				return False

		return True

	def isLeaf(self):
		return False

	def predict(self, X):
		if X[self.attribute] == 0:
			return self.children[0].predict(X)
		else:
			return self.children[1].predict(X)


class Leaf: 

	def __init__(self, label):
		self.label = label[0]

	def label(self):
		return self.label

	def depth(self):
		return 1

	def isLeaf(self):
		return True

	def printTree(self):
		print self.label

	def predict(self, X):
		return self.label