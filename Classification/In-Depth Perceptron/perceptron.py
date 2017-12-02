from numpy.random import random_sample
import numpy as np
import  os

class Perceptron(object):

    def load_data(self, file_name):
        # Array of names
        X = []

        # Array of labels (+1, -1)
        Y = []
        for line in open(os.path.join(os.path.dirname(__file__), file_name)):
            tokens = line.split()
            label = tokens[0]
            Y.append(float(label))
            line = line[3:]
            tokens = line.split(' ')

            features = np.zeros(68)

            for feature in tokens:
                feature_split = feature.split(':')
                index = int(feature_split[0]) - 1
                value = float(feature_split[1])
                features[index] = value

            X.append(features)

        return X, Y

    def score(self, perceptron, X, Y):
        correct = 0
        for (x, y) in zip(X, Y):
            if perceptron.predict(x) == y:
                correct += 1

        return correct / float(len(Y))

    def score_averaged(self, perceptron, X, Y):
        correct = 0
        for (x, y) in zip(X, Y):
            if perceptron.predict_avreaged(x) == y:
                correct += 1

        return correct / float(len(Y))

    def __init__(self, feature_size, learning_rate):
        np.random.seed(19)
        self.w = np.random.uniform(-0.1, 0.1, feature_size)
        self.learning_rate = learning_rate
        self.bias = np.random.uniform(-0.1, 0.1, 1)
        self.bias_2 = np.random.uniform(-0.1, 0.1, 1)
        self.a = np.random.uniform(-0.1, 0.1, feature_size)
        self.updates = 0

        # epoch plotting
        self.epocCount = []
        self.devSetAccuracy = []
        self.devSetAccuracyAvg = []

    def predict(self, x):
        predict_y = np.dot(self.w, x) + self.bias
        if predict_y >= 0:
            return 1
        else:
            return -1

    def predict_avreaged(self, x):
        predict_y = np.dot(self.a, x) + self.bias_2
        if predict_y >= 0:
            return 1
        else:
            return -1


    def train_dynamic_learning_rate(self, X, Y, epochs=10, margin=0.0):

        X, Y = self.load_data('phishing.dev')
        init_learning_rate = self.learning_rate
        t = 0

        for i in range(0, epochs):
            self.epocCount.append(i + 1)
            for (x, y) in zip(X, Y):
                predict_y = y*(np.dot(self.w, x) + self.bias)
                if predict_y < margin:
                    # update weights
                    self.w = self.w + self.learning_rate * y * x
                    # update bias
                    self.bias = self.bias + float(self.learning_rate * y)
                    # update learning rate
                    self.learning_rate = init_learning_rate / float((1 + t))
                    self.updates += 1
                t += 1.0

            self.devSetAccuracy.append(self.score(self, X, Y))

    def train_simple(self, X, Y, epochs=10):
        X, Y = self.load_data('phishing.dev')
        for i in range(0, epochs):
            self.epocCount.append(i + 1)
            for (x, y) in zip(X, Y):
                predict_y = y*(np.dot(self.w, x) + self.bias)
                if predict_y < 0:
                    # update weights
                    self.w = self.w + self.learning_rate * y * x
                    # update bias
                    self.bias = self.bias + float(self.learning_rate * y)
                    self.updates += 1

                self.a = self.a + self.w
                self.bias_2 = self.bias_2 + self.bias

            self.devSetAccuracy.append(self.score(self, X, Y))
            self.devSetAccuracyAvg.append(self.score_averaged(self, X, Y))

