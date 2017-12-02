import os
import numpy as np
from perceptron import Perceptron
import itertools
import matplotlib.pyplot as plt

def load_data(file_name):
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

        features = np.zeros(16)

        for feature in tokens:
            feature_split = feature.split(':')
            index = int(feature_split[0]) - 1
            value = float(feature_split[1])
            features[index] = value

        X.append(features)

    return X, Y

def merged_excluding(X_trains, Y_trains, index):
    X = []
    Y = []

    for i in range(0, len(X_trains)):
        if i != index:
            X = X + X_trains[i]
            Y = Y + Y_trains[i]

    return X, Y


XCV_1, YCV_1 = load_data('CVSplits/training00.data')
XCV_2, YCV_2 = load_data('CVSplits/training01.data')
XCV_3, YCV_3 = load_data('CVSplits/training02.data')
XCV_4, YCV_4 = load_data('CVSplits/training03.data')
XCV_5, YCV_5 = load_data('CVSplits/training04.data')

XCV = [XCV_1, XCV_2, XCV_3, XCV_4, XCV_5]
YCV = [YCV_1, YCV_2, YCV_3, YCV_4, YCV_5]
CV_Labels = ['trainingdata00', 'trainingdata01', 'trainingdata02', 'trainingdata03', 'trainingdata04']

def score(perceptron, X, Y):
    correct = 0
    for (x, y) in zip(X, Y):
        if perceptron.predict(x) == y:
            correct += 1

    return correct / float(len(Y))

def score_averaged(perceptron, X, Y):
    correct = 0
    for (x, y) in zip(X, Y):
        if perceptron.predict_avreaged(x) == y:
            correct += 1

    return correct / float(len(Y))

def plot(perceptron, title, avg=False):
    x = perceptron.epocCount
    y = perceptron.devSetAccuracy
    if avg:
        y = perceptron.devSetAccuracyAvg

    # plt.xlabel('Epoch Count')
    # plt.ylabel('Dev set accuracy')
    # plt.suptitle(title)
    # plt.plot(x, y, 'o-', color='g')
    # plt.savefig('%s.pdf' % title)
    # plt.clf()

def experiment1():
    # Simple perceptron
    learning_rates = [1, 0.1, 0.01]

    for learning_rate in learning_rates:
        count = 0.0
        for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
            X_test, Y_test = merged_excluding(XCV, YCV, i)

            # Simple perceptron
            perceptron = Perceptron(len(X[0]), learning_rate)
            perceptron.train_simple(X, Y, 10)
            accuracy = score(perceptron, X, Y)
            print 'Simple perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (data_label, accuracy, learning_rate)

            # Dynamic learning rate
            perceptron = Perceptron(len(X[0]), learning_rate)
            perceptron.train_dynamic_learning_rate(X, Y, epochs=10)
            accuracy = score(perceptron, X_test, Y_test)
            print 'Dynamic perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (data_label, accuracy, learning_rate)


            # Averaged perceptron
            perceptron = Perceptron(len(X[0]), learning_rate)
            perceptron.train_simple(X, Y, 10)
            accuracy = score_averaged(perceptron, X, Y)
            print 'Averaged perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (data_label, accuracy, learning_rate)

        print ''
        print 'Average for learning rate = %.2f is %.3f' % (learning_rate, (count / float(len(XCV)) * 100))


    margins = [1, 0.1, 0.01]

    maxValue = 0.0
    maxLearningRate = 0.0
    maxMargin = 0.0
    for learning_rate, margin in itertools.product(learning_rates, margins):
        count = 0.0
        for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
            X_test, Y_test = merged_excluding(XCV, YCV, i)

            # Margin Perceptron
            perceptron = Perceptron(len(X[0]), learning_rate)
            perceptron.train_dynamic_learning_rate(X, Y, 10, margin)
            accuracy = score(perceptron, X, Y)
            count += accuracy
            print 'margin perceptron training on %s got an accuracy of %.3f with learning rate = %.2f and margin = %.2f' %(data_label, accuracy, learning_rate, margin)

            if count > maxValue:
                maxValue = count
                maxLearningRate = learning_rate
                maxMargin = margin

            # print 'Average for learning rate = %.2f and margin = %.2f is %.3f' % (learning_rate, margin, (count / float(len(XCV)) * 100))
    print 'Best learning rate for margin perceptron = %.2f' % maxLearningRate
    print 'Best margin for margin perceptron = %.2f' % maxMargin


    print '** CV ACCURACY **'
    # 2. CV accuracy
    for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
        X_test, Y_test = merged_excluding(XCV, YCV, i)

        # Simple perceptron
        perceptron = Perceptron(len(X[0]), 0.1)
        perceptron.train_simple(X, Y, 10)
        accuracy = score(perceptron, X, Y)
        # print 'Simple perceptron training on %s got an accuracy of %.3f' % (
        # data_label, accuracy)

        # Dynamic learning rate
        perceptron = Perceptron(len(X[0]), 1.0)
        perceptron.train_dynamic_learning_rate(X, Y, epochs=10)
        accuracy = score(perceptron, X_test, Y_test)
        # print 'Dynamic perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (
        # data_label, accuracy, learning_rate)

        # Margin perceptron
        perceptron = Perceptron(len(X[0]), 1)
        perceptron.train_dynamic_learning_rate(X, Y, epochs=10, margin=0.1)
        accuracy = score(perceptron, X, Y)
        # print 'Margin perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (
        # data_label, accuracy, learning_rate)


        # Averaged perceptron
        perceptron = Perceptron(len(X[0]), 0.10)
        perceptron.train_simple(X, Y, 10)
        accuracy = score_averaged(perceptron, X, Y)
        print 'Averaged perceptron training on %s got an accuracy of %.3f with learning rate = %.2f' % (
        data_label, accuracy, learning_rate)

    # 3. updates on phising.test and 4. Dev set accuracy and 5. test data accuracy
    X, Y = load_data('phishing.train')
    X_dev, Y_dev = load_data('phishing.dev')
    X_test, Y_test = load_data('phishing.test')

    perceptron = Perceptron(len(X[0]), 0.1)
    perceptron.train_simple(X, Y, 19)
    print 'Simple perceptron performed %d updates' % perceptron.updates
    print 'Simple perceptron accuracy on dev set %.3f' % score(perceptron, X_dev, Y_dev)
    print 'Simple perceptron accuracy on test set %.3f\n' % score(perceptron, X_test, Y_test)

    perceptron = Perceptron(len(X[0]), 1.0)
    perceptron.train_dynamic_learning_rate(X, Y, 16)
    print 'Dynamic perceptron performed %d updates' % perceptron.updates
    print 'Dynamic perceptron accuracy on dev set %.3f' % score(perceptron, X_dev, Y_dev)
    print 'Dynamic perceptron accuracy on test set %.3f\n' % score(perceptron, X_test, Y_test)

    perceptron = Perceptron(len(X[0]), 1.0)
    perceptron.train_dynamic_learning_rate(X, Y, 19, 0.01)
    print 'Margin perceptron performed %d updates' % perceptron.updates
    print 'Margin perceptron accuracy on dev set %.3f' % score(perceptron, X_dev, Y_dev)
    print 'Margin perceptron accuracy on test set %.3f\n' % score(perceptron, X_test, Y_test)

    perceptron = Perceptron(len(X[0]), 0.10)
    perceptron.train_simple(X, Y, 11)
    print 'Averaged perceptron performed %d updates' % perceptron.updates
    print 'Averaged perceptron accuracy on dev set %.3f' % score_averaged(perceptron, X_dev, Y_dev)
    print 'Averaged perceptron accuracy on test set %.3f\n' % score_averaged(perceptron, X_test, Y_test)

    # Plotting
    perceptron = Perceptron(len(X[0]), 0.1)
    perceptron.train_simple(X, Y, 20)
    plot(perceptron, 'Simple Perceptron Epoch Accuracy')

    perceptron = Perceptron(len(X[0]), 1.0)
    perceptron.train_dynamic_learning_rate(X, Y, 20)
    plot(perceptron, 'Dynamic Perceptron Epoch Accuracy')

    perceptron = Perceptron(len(X[0]), 1.0)
    perceptron.train_dynamic_learning_rate(X, Y, 20, 0.01)
    plot(perceptron, 'Margin Perceptron Epoch Accuracy')

    perceptron = Perceptron(len(X[0]), 0.10)
    perceptron.train_simple(X, Y, 20)
    plot(perceptron, 'Averaged Perceptron Epoch Accuracy', True)


def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def number2():
    X_train, Y_train = load_data('phishing.train')
    majority_label = find_majority(Y_train)
    print 'Majority label in phising.train = %d' % majority_label[0]

    X_test, Y_train = load_data('phishing.test')
    X_dev, Y_dev = load_data('phishing.dev')
    test_score = 0
    dev_score = 0

    for (y_test, y_dev) in zip(Y_train, Y_dev):
        if y_test == majority_label[0]:
            test_score += 1
        if y_dev == majority_label[0]:
            dev_score += 1

    print 'Majority baseline for test set = %.3f' % (test_score / float(len(Y_train)))
    print 'Majority baseline for dev set = %.3f' % (dev_score / float(len(Y_train)))


print 'Majority baseline'
number2()
print ''
experiment1()
print 'Done'
