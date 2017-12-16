import numpy as np
from SVM import *
import random
import math

def shuffle(X, Y):
    XY = list(zip(X, Y))
    random.shuffle(XY)
    X, Y = zip(*XY)
    return np.array(X), np.array(Y)

def log2(x):
    if x == 0:
        return 0
    negative = False
    if x < 0.0:
        negative = True

    value = math.log(abs(x), 2)

    if negative:
        value = -value

    return value

def SGD(X, Y, feature_count, epochs=10, learning_rate=10**-2, tradeoff=10**2):
    w = np.zeros(feature_count)

    for epoch in range(0, epochs):
        X_shuffle, Y_shuffle = shuffle(X, Y)
        step_count = 0
        for (x, y) in zip(X_shuffle, Y_shuffle):
            gamma_t = learning_rate / (1 + (learning_rate * step_count / tradeoff))
            w -= gamma_t * ( (-y * x) / (1 + math.exp(y * log2(np.dot(w, x)))) + ((2 * w) / tradeoff))
            step_count += 1

    return w


def predict(w, x):
    y_prime = np.dot(w, x)
    if y_prime >= 0:
        return 1
    else:
        return 0

def get_predictions(w, X):
    predictions = []
    for x in X:
        predictions.append(predict(w, x))

    return predictions

def run_logistic_regression(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
    print '************* REGULAR LOGISTIC REGRESSION *************'

    feature_count = 0
    for example in X_train:
        feature_count = max(feature_count, max(example))

    for example in X_test:
        feature_count = max(feature_count, max(example))

    X_train, Y_train = convert_collection(X_train, Y_train, feature_count)
    for i in range(0, len(XCV)):
        new_x, new_y = convert_collection(XCV[i], YCV[i], feature_count)
        XCV[i] = new_x
        YCV[i] = new_y

    learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
    tradeoffs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]

    best_learning_rate = 0.10
    best_tradeoff = 100
    best_avg = 0.0


    for learning_rate in learning_rates:
        for tradeoff in tradeoffs:
            current_avg = 0.0
            for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
                X_cv, Y_cv = merged_excluding(XCV, YCV, i)
                weight_vector = SGD(X_cv, Y_cv, feature_count, learning_rate=learning_rate, tradeoff=tradeoff)
                predictions = get_predictions(weight_vector, XCV[i])
                accuracy = score(predictions, YCV[i])
                current_avg += float(accuracy)
                print 'Logistic Regression: Accuracy = %.3f, Learning rate = %.3f, Trade off = %.3f on test set %s' % (accuracy, learning_rate, tradeoff, data_label)

            print 'Average CV accuracy for LR = %.3f and tradeoff = %.3f accuracy = %.3f' % (learning_rate, tradeoff, current_avg / len(XCV))            
            if current_avg / len(XCV) > best_avg:
                current_avg = current_avg / len(XCV)
                best_avg = current_avg
                best_learning_rate = learning_rate
                best_tradeoff = tradeoff

    print 'Best hyperparams: learning rate = %.2f, tradeoff = %.2f with an average of %.3f' % (best_learning_rate, best_tradeoff, best_avg)

    X_train, Y_train = convert_collection(X_train, Y_train, feature_count)
    weight_vector = SGD(X_train, Y_train, feature_count, learning_rate=best_learning_rate, tradeoff=best_tradeoff)
    predictions = get_predictions(weight_vector, X_train)
    accuracy = score(predictions, Y_train)
    print 'Logistic Regression: Accuracy on train = %.3f, with best hyperparams' % (accuracy)

    X_test, Y_test = convert_collection(X_test, Y_test, feature_count)
    # weight_vector = SGD(X_test, Y_test, feature_count, learning_rate=best_learning_rate, tradeoff=best_tradeoff)
    predictions = get_predictions(weight_vector, X_test)
    accuracy = score(predictions, Y_test)
    print 'Logistic Regression: Accuracy on test = %.3f, with best hyperparams' % (accuracy)
