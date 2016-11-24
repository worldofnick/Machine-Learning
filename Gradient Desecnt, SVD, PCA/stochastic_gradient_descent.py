# Nick Porter, CS 4964 - Math for Data HW 4, Q1 (b), University of Utah
import pandas as pandas
import statsmodels.api as sm
import numpy as np
import random

# Load the data
x = pandas.read_csv('D4.csv', usecols = [0,1,2])
y = pandas.read_csv('D4.csv', usecols = [3])
x.columns = ['a_1', 'a_2', 'a_3']
y.columns = ['y']

def stochastic_gradient_descent(learning_rate, iterations, x, y):
	x = x.as_matrix()
	y = y.as_matrix()

	alphas = np.zeros(4)
	# Use every data point at least once
	data_points = np.arange(140)
	random.shuffle(data_points)

	for i in range(1, iterations):
		# Our random data point
		data_point = data_points[i]
		hypothesis = evaluate(x[data_point], alphas)
		# x_i - y
		loss = hypothesis - y[data_point]
		cost = loss**2 

		# Compute gradient
		gradient = np.zeros(4)
		gradient[0] = loss
		gradient[1] = loss * x[data_point][0]
		gradient[2] = loss * x[data_point][1]**2
		gradient[3] = loss * x[data_point][2]**3

		gradient = gradient * 2
		alphas = alphas - learning_rate * gradient

		print("Iteration %d | f(x): %f" % (i, hypothesis))
		print("Iteration %d | Cost: %f" % (i, cost))
		print("Iteration %d | Alpha: %s" % (i, alphas))

def evaluate(x, alphas):
	return alphas[0] + (alphas[1] * x[0]) + (alphas[2] * x[1]) + (alphas[3] * x[2])

stochastic_gradient_descent(0.005, 40, x, y)

