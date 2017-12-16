from logisticregression import *
from SVM import *
from svm_trees import data_set_from_roots
from bagged_forest import build_roots, setToN
import copy

import dt

def run_logistic_regression_trees(X_train, Y_train, X_test, Y_test, XCV, YCV, CV_Labels):
    print '******************* LOGISTIC REGRESSION TREES *****************'
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
    tradeoffs = [10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
    attrs = setToN(feature_count)

    best_learning_rate = 0.01
    best_tradeoff = 10000
    best_avg = 0.0
    depth = 3

    for learning_rate in learning_rates:
        for tradeoff in tradeoffs:
            current_avg = 0.0
            for (X, Y, data_label, i) in zip(XCV, YCV, CV_Labels, range(0, len(XCV))):
                X_cv, Y_cv = merged_excluding(XCV, YCV, i)
                roots = build_roots(X_cv, Y_cv, attrs, depth)
                transformed_X = data_set_from_roots(roots, X_cv)
                weight_vector = SGD(transformed_X, Y_cv, len(roots), learning_rate=learning_rate, tradeoff=tradeoff)
                predictions = get_predictions(weight_vector, data_set_from_roots(roots, XCV[i]))
                accuracy = score(predictions, YCV[i])
                current_avg += float(accuracy)
                dt.zeroed.clear()
                print 'Logistic Regression Over Trees: Accuracy = %.5f, Learning rate = %.3f, Trade off = %.3f on test set %s' % (accuracy, learning_rate, tradeoff, data_label)
                
        
            print 'Average CV accuracy for LR over trees = %.5f and tradeoff = %.3f accuracy = %.3f' % (learning_rate, tradeoff, current_avg / len(XCV))
            if current_avg / len(XCV) > best_avg and i == 4:
                current_avg = current_avg / len(XCV)
                best_avg = current_avg
                best_learning_rate = learning_rate
                best_tradeoff = tradeoff



    roots = build_roots(X_train, Y_train, attrs, 3)
    transformed_X = data_set_from_roots(roots, X_train)
    weight_vector = SGD(transformed_X, Y_train, len(roots), learning_rate=best_learning_rate, tradeoff=best_tradeoff)
    predictions = get_predictions(weight_vector, transformed_X)
    accuracy = score(predictions, Y_train)
    print 'LR Trees train accuracy with best hyperparams = %.3f' % accuracy

    transformed_X = data_set_from_roots(roots, X_test, debug=False)
    predictions = get_predictions(weight_vector, transformed_X)
    accuracy = score(predictions, Y_test)
    print 'LR Trees test accuracy with best hyperparams = %.3f' % accuracy
    
