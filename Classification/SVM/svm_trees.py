from SVM import SVM
from SVM import *
from bagged_forest import build_roots, setToN
import numpy as np
import dt
import copy

def data_set_from_roots(roots, X, debug=False):
    transformed = []

    for x in copy.deepcopy(X):
        predictions = []
        for root in roots:
            prediction = root.predict(x)
            predictions.append(prediction)

        transformed.append(np.array(predictions))

    return np.array(transformed)

def run_svm_trees(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
    print '******************* SVM TREES *****************'
    feature_count = 0
    for example in X_train:
        feature_count = max(feature_count, max(example))

    for example in X_test:
        feature_count = max(feature_count, max(example))

    X_train, Y_train = convert_collection(X_train, Y_train, feature_count)
    X_test, Y_test = convert_collection(X_test, Y_test, feature_count)

    for i in range(0, len(XCV)):
        new_x, new_y = convert_collection(XCV[i], YCV[i], feature_count)
        XCV[i] = new_x
        YCV[i] = new_y


    learning_rates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
    trade_offs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
    depths = [3]
    depth = 3
    best_avg = 0.0
    best_learning_rate = 0.1
    best_trade_off = 10000
    best_depth = depths[0]
    attrs = setToN(feature_count)

    for learning_rate in learning_rates:
        for trade_off in trade_offs:
            current_avg = 0.0
            for depth in depths:
                for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
                    X_cv, Y_cv = merged_excluding(XCV, YCV, i)
                    roots = build_roots(X_cv, Y_cv, attrs, depth)
                    transformed_X = data_set_from_roots(roots, X_cv)
                    weight_vector = compute_weights(transformed_X, Y_cv, len(roots), learning_rate=learning_rate, tradeoff=trade_off)
                    predictions = get_predictions(weight_vector, data_set_from_roots(roots, XCV[i]))
                    accuracy = score(predictions, YCV[i])
                    current_avg += accuracy
                    dt.zeroed.clear()
                    print 'SVM Trees: Accuracy = %.3f, Learning rate = %.5f, Trade off = %.3f, Depth = %d on test set %s' % (accuracy, learning_rate, trade_off, depth, data_label)

        
                    if current_avg / len(XCV) > best_avg and i == 4:
                        current_avg = current_avg / len(XCV)
                        best_avg = current_avg
                        best_learning_rate = learning_rate
                        best_trade_off = trade_off
                        best_depth = depth


    print 'Best learning rate = %.5f and trade off = %.5f and depth = %d with an average accuracy of %.3f' % (best_learning_rate, best_trade_off, best_depth, best_avg)

    roots = build_roots(X_train, Y_train, attrs, 3)
    transformed_X = data_set_from_roots(roots, X_train)
    weight_vector = compute_weights(transformed_X, Y_train, len(roots), learning_rate=best_learning_rate, tradeoff=best_trade_off)
    predictions = get_predictions(weight_vector, transformed_X)
    accuracy = score(predictions, Y_train)
    print 'SVM Trees train accuracy with best hyperparams = %.3f' % accuracy

    predictions = get_predictions(weight_vector, data_set_from_roots(roots, X_test))
    accuracy = score(predictions, Y_test)
    print 'SVM Trees test accuracy with best hyperparams = %.3f' % accuracy
