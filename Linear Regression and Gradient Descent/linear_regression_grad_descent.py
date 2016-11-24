# Nick Porter, CS 4964 - Math for Data HW 3, University of Utah
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from numpy import linalg as LA

# 1c
def poly_regression(x, y, degree):
	coefs = np.polyfit(x.values.flatten(), y.values.flatten(), degree)
	p = np.poly1d(coefs)
	print 'Degree:'
	print degree
	print coefs

def sse(X_test, y_test, p):
	sse = 0
	for i in range(0, len(X_test)):
		xValue = X_test.iloc[i][0]
		yValue = y_test.iloc[i][0]
		sse = sse + (p(xValue) - yValue)**2
	return sse

def pick_best_model(x, y):
	sums = [0,0,0,0,0]
	count = 0
	for i in range(1, 10001):
		count = count + 1
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
		for j in range(1, 6):
			coefs = np.polyfit(x.values.flatten(), y.values.flatten(), j)
			p = np.poly1d(coefs)
			sum_sqaured_error = sse(X_test, y_test, p)
			sums[j - 1] = sums[j - 1] + sum_sqaured_error


	for i in range(0, len(sums)):
		print i + 1 # degree
		print sums[i] / count # average sse


x = pandas.read_csv('D3.csv', usecols = [0])
y = pandas.read_csv('D3.csv', usecols = [3])

# 1a
m, b = np.polyfit(x.values.flatten(), y.values.flatten(), 1) # degree 1 for a linear line
p = np.poly1d([m, b])

# 1b
# Split data randomly into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

# Build our new model from the training data
# trainM, trainB = np.polyfit(X_train.values.flatten(), y_train.values.flatten(), 1)
# p = np.poly1d([trainM, trainB])
# sum_sqaured_error = sse(X_test, y_test, p)

# 1c
# for i in range(1,6):
# 	plot_poly(X_train, y_train, X_test, y_test, i)

# 1d
#pick_best_model(x, y)

# 2a
x3 = pandas.read_csv('D3.csv', usecols = [0,1,2])
x3 = sm.add_constant(x3)
model = sm.OLS(y,x3).fit()
#print model.params
# print 'f(x_1, x_2, x_3) = %sx_1 + %sx_2 + %sx_3 + %s' % (model.params[1], model.params[2], model.params[3], model.params[0])
# #print (1 * model.params[1]) + (1 * model.params[2]) + (1 * model.params[3]) + model.params[0]
# print 'f(1, 1, 1) = %s' % ((1 * model.params[1]) + (1 * model.params[2]) + (1 * model.params[3]) + model.params[0])
# print 'f(2, 0, 4) = %s' % ((2 * model.params[1]) + (0 * model.params[2]) + (4 * model.params[3]) + model.params[0])
# print 'f(3, 2, 1) = %s' % ((3 * model.params[1]) + (2 * model.params[2]) + (1 * model.params[3]) + model.params[0])

# #2b
# X_train, X_test, y_train, y_test = train_test_split(x3, y, test_size=0.10)

# sse = 0
# for i in range(0, len(X_test)):
# 	xValue_1 = X_test.iloc[i][1]
# 	xValue_2 = X_test.iloc[i][2]
# 	xValue_3 = X_test.iloc[i][3]
	
# 	yValue = y_test.iloc[i][0]
# 	sse = sse + ((xValue_1 * model.params[1] + xValue_2 * model.params[2] + xValue_3 * model.params[3] + model.params[0]) - yValue)**2
# print 'SSE: %s' % (sse)


#3
def function1(x, y):
	return (x - 2)**2 + (y - 3)**2

def function1_grad(x, y):
	dx = 2.0 * (x - 2.0)
	dy = 2.0 * (y - 3.0)
	return np.array([dx, dy])

def function2(x, y):
	return (1 - (y - 3))**2 + 20 * ((x + 3) - (y - 3)**2)**2

def function2_grad(x,y):
	dx = 40.0*(3.0 + x -(-3.0 + y)**2.0)
	dy = 2.0*(-4.0-40.0*(3.0+x-(-3.0+y)**2.0)*(-3.0 + y) + y)
	return np.array([dx, dy])

def gradient_descent(gradf, x, y, learning_rate, iterations):

	v_init = np.array([x, y])
	values = np.zeros([iterations, 2])
	values[0, :] = v_init
	v = v_init

	for i in range(1, iterations):
		print v
		v = v - learning_rate * gradf(v[0], v[1])
		values[i, :] = v

	print LA.norm(gradf(v[0], v[1]))

# print 'f_1(x)'
# gradient_descent(function1_grad, 0, 0, 0.5, 10)
# print 'f_2(x)'
# gradient_descent(function2_grad, 0, 0, 0.5, 10)

print 'f_1(x) run 2'
gradient_descent(function1_grad, 5, 5, 0.01, 100)
print 'f_2(x) run 2'
gradient_descent(function2_grad, 6, 6, 0.0007, 100)


# # 1a, 1b Plots
# plt.plot(x, y, '.', color='green')
# plt.plot(x, m*x + b, '-', color='blue')
# #plt.title("y = %.2fx + %.2f" % (m, b))
# plt.plot(X_train, trainM*X_train + trainB, '-', color='red')
# #plt.title("Blue is actual, red is after CV. SSE = %.2f" % (sum_sqaured_error))
# plt.show()
