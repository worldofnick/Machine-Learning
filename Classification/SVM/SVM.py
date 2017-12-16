import numpy as np
import random, math

class SVM:


    def __init__(self):
        print 'not yet implemented'

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

def compute_weights(X, Y, feature_count, epochs=10, learning_rate=10**-2, tradeoff=10**2):
    w = np.zeros(feature_count)
    
    for epoch in range(0, epochs):
        X_shuffle, Y_shuffle = shuffle(X, Y)
        step_count = 0
        for (x, y) in zip(X_shuffle, Y_shuffle):
            learning_rate_t = learning_rate / (1 + step_count)
            if y * np.dot(w, x) <= 1:
                w = np.add((1 - learning_rate_t) * w, learning_rate_t * tradeoff * y * x)    
            else:
                w = (1 - learning_rate_t) * w
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


def convert_collection(X, Y, feature_count):
    X_c = []
    Y_c = []
    count = 0

    for (x, y) in zip(X, Y):
        x_c, y_c = convert(x, y, feature_count)
        X_c.append(x_c)
        Y_c.append(y_c)
        count += 1

    return np.array(X_c), np.array(Y_c)

def convert(x, y, feature_count):
    arr = np.zeros(feature_count)
    if type(x) == np.ndarray:
        return x, y

    for key, value in x.iteritems():
        arr[key - 1] = value

    return arr, y

def merged_excluding(X_trains, Y_trains, index):
    X = []
    Y = []

    for i in range(0, len(X_trains)):
        if i != index:
            X = X + list(X_trains[i])
            Y = Y + list(Y_trains[i])

    return np.array(X), np.array(Y)

def score(predictions, Y):
    correct = 0
    for (p, y) in zip(predictions, Y):
        if p == 0:
            p = -1
        if p == y:
            correct += 1

    return correct / float(len(Y))

def run_svm(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
    print '************* REGULAR SVM *************'
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

    learning_rates = [10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    tradeoffs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]

    best_learning_rate = 0.10
    best_tradeoff = 100
    best_avg = 0.0

    for learning_rate in learning_rates:
        for trade_off in tradeoffs:
            pair_avg = 0.0
            for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
                X_cv, Y_cv = merged_excluding(XCV, YCV, i)
                w = compute_weights(X_cv, Y_cv, feature_count, learning_rate=learning_rate, tradeoff=trade_off)
                predictions = get_predictions(w, XCV[i])
                accuracy = score(predictions, YCV[i])
                pair_avg += accuracy
                print 'SVM: Accuracy = %.3f, Learning rate = %.3f, Trade off = %.3f on test set %s' % (accuracy, learning_rate, trade_off, data_label)
            
            if pair_avg / len(XCV) > best_avg and i == len(XCV) - 1:
                pair_avg = pair_avg / len(XCV)
                best_avg = pair_avg
                best_learning_rate = learning_rate
                best_trade_off = trade_off

    print 'Best learning rate = %.3f and trade off = %.3f with an average accuracy of %.3f' % (best_learning_rate, best_trade_off, best_avg)

    w = compute_weights(X_train, Y_train, feature_count, learning_rate=best_learning_rate, tradeoff=best_tradeoff)
    predictions = get_predictions(w, X_train)
    print 'SVM train accuracy with best hyperparams = %.3f' % score(predictions, Y_train)

    X_test, Y_test = convert_collection(X_test, Y_test, feature_count)
    predictions = get_predictions(w, X_test)
    print 'SVM test accuracy with best hyperparams = %.3f' % score(predictions, Y_test)
    print '************* END REGULAR SVM *************'
