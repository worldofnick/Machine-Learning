# Nick Porter, CS 4964 - Math for Data HW 4, Q1 (a), University of Utah
import pandas as pandas
import statsmodels.api as sm
import numpy as np

# Load the data
x = pandas.read_csv('D4.csv', usecols = [0,1,2])
y = pandas.read_csv('D4.csv', usecols = [3])

def batch_gradient_descent(learning_rate, iterations, x, y):
	x = x.as_matrix()
	y = y.as_matrix()

	alphas = np.zeros(4)

	for i in range(1, iterations):
		gradient = np.zeros(4)
		total_hypo = 0
		total_cost = 0

		# Compute the gradient for all data points
		for data_point in range(0, 149):
			hypothesis = evaluate(x[data_point], alphas)
			
			# x_i - y
			loss = hypothesis - y[data_point]
			cost = loss**2 

			total_hypo += hypothesis
			total_cost += cost

			temp_gradient = np.zeros(4)
			temp_gradient[0] = loss
			temp_gradient[1] = loss * x[data_point][0]
			temp_gradient[2] = loss * x[data_point][1]**2
			temp_gradient[3] = loss * x[data_point][2]**3
			gradient += temp_gradient

		gradient *= 2
		alphas = alphas - learning_rate * gradient
		print("Iteration %d | Average f(x): %f" % (i, total_hypo/149))
		print("Iteration %d | Average Cost: %f" % (i, total_cost/149))
		print("Iteration %d | Alpha: %s" % (i, alphas))


def evaluate(x, alphas):
	return alphas[0] + (alphas[1] * x[0]) + (alphas[2] * x[1]) + (alphas[3] * x[2])

batch_gradient_descent(0.00003, 15, x, y)

