from SVM import convert_collection, merged_excluding, score
from collections import Counter
import numpy
from scipy.sparse import csr_matrix 
import math

import warnings
warnings.filterwarnings("ignore")

class NB(object):
    def __init__(self, smoothing_term=0.5):
        self.smoothing_term = smoothing_term
        self.proportion_positive = None
        self.proportion_negative = None
        self.plus_plus = None
        self.neg_plus = None
        self.plus_neg = None
        self.neg_neg = None
        

    def train(self, X, Y):
        X = csr_matrix(X)
        
        positive_y_matrix = csr_matrix(numpy.expand_dims(numpy.where(Y == 1.0, 1.0, 0.0), axis=1))
        negative_y_matrix = csr_matrix(numpy.expand_dims(numpy.where(Y == -1.0, 1.0, 0.0), axis=1))
        smoothed = 2.0 * self.smoothing_term

        self.plus_plus = numpy.divide(numpy.squeeze(numpy.array(X.multiply(positive_y_matrix).sum(axis=0))) + (smoothed), numpy.sum(numpy.where(Y == 1, 1, 0)) + (smoothed))
        self.neg_plus = numpy.divide(numpy.squeeze(numpy.array(X.multiply(negative_y_matrix).sum(axis=0))) + (smoothed), numpy.sum(numpy.where(Y == -1, 1, 0)) + (smoothed))

        self.plus_neg = 1.0 - self.plus_plus
        self.neg_neg = 1.0 - self.neg_plus
        self.log_all(Y)

    def get_predictions(self, X):
        feature_count = X.shape[0]
        predictions = []

        for index in range(X.shape[0]):
            if (numpy.sum(numpy.where(X[index] == 1.0, self.plus_plus, self.plus_neg)) + self.proportion_positive) >= (numpy.sum(numpy.where(X[index] == 1.0, self.neg_plus, self.neg_neg)) + self.proportion_negative):
                predictions.append(1.0)
            else:
                predictions.append(-1.0)

        return predictions


    def log_all(self, Y):
        self.plus_plus = numpy.log2(self.plus_plus)
        self.neg_plus = numpy.log2(self.neg_plus)
        self.plus_neg = numpy.log2(self.plus_neg)
        self.neg_neg = numpy.log2(self.neg_neg)
        self.proportion_positive = math.log(numpy.sum(numpy.where(Y == 1, 1.0, 0.0)) / float(len(Y)), 2)
        self.proportion_negative = math.log(1.0 - numpy.sum(numpy.where(Y == 1, 1.0, 0.0)) / float(len(Y)), 2)

def get_feature_count(X):
    max_value = 0
    for example in X:
        max_value = max(max_value, max(example))
    return max_value

def run_naive_bayes(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
    print '******************* NAIVE BAYES *****************'
    feature_count = max(get_feature_count(X_train), get_feature_count(X_test))

    X_train, Y_train = convert_collection(X_train, Y_train, feature_count)
    X_test, Y_test = convert_collection(X_test, Y_test, feature_count)

    for i in range(0, len(XCV)):
        new_x, new_y = convert_collection(XCV[i], YCV[i], feature_count)
        XCV[i] = new_x
        YCV[i] = new_y

    smoothing_terms = [2, 1.5, 1.0, 0.5]

    best_smoothing_term = smoothing_terms[0]
    best_avg = 0.0

    for smoothing_term in smoothing_terms:
        current_avg = 0.0
        for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
            X_cv, Y_cv = merged_excluding(XCV, YCV, i)
            nb = NB(smoothing_term=smoothing_term)
            nb.train(X_cv, Y_cv)
            accuracy = score(nb.get_predictions(XCV[i]), YCV[i])
            current_avg += accuracy
            print 'Naive Bayes: Accuracy = %.3f, Smoothing term = %.3f test set %s' % (accuracy, smoothing_term, data_label)
            if current_avg / len(XCV) > best_avg and i == 4:
                current_avg = current_avg / len(XCV)
                best_avg = current_avg
                best_smoothing_term = smoothing_term



    print 'Best smoothing term = %.3f with avg accuracy of %.3f' % (best_smoothing_term, best_avg)
    nb = NB(smoothing_term=best_smoothing_term)
    nb.train(X_train, Y_train)
    accuracy = score(nb.get_predictions(X_train), Y_train)
    print 'Naive Bayes train accuracy with best hyperparams = %.3f' % accuracy

    accuracy = score(nb.get_predictions(X_test), Y_test)
    print 'Naive Bayes test accuracy with best hyperparams = %.3f' % accuracy

    print '************* END Naive Bayes *************'
