import random
import timeit
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def couponExperiment(n):
	seen = set()
	iterations = 0
	while len(seen) != n:
		seen.add(random.randint(1, n))
		iterations += 1
	return iterations

def required_trials(m, n):
	required_trials = []
	for i in range(0, m):
		required_trials.append(couponExperiment(n))
	return required_trials

n = 200
# (a)
print 'Q2 (a): Number of trials required %s' % couponExperiment(n) 

# (b)
m = 300
required_trials_arr = required_trials(m, n)
y = [1.0/x for x in required_trials_arr]
data = zip(required_trials_arr, y)
# Choose how many bins you want here
num_bins = m

# Use the histogram function to bin the data
counts, bin_edges = np.histogram(data, bins=num_bins, normed=False)

# Now find the cdf
cdf = np.cumsum(counts)
cdf = preprocessing.MinMaxScaler().fit_transform(cdf)

# And finally plot the cdf
# plt.plot(bin_edges[1:], cdf, linewidth=2.0)

# plt.xlabel('trials required (k)')
# plt.ylabel('fraction of success')
# plt.title('Coupon Collector')
# plt.show()

# (c)
print sum(required_trials_arr)
print m
print 'Q1 (c): Empirical Estimate %f' % (sum(required_trials_arr) / m)

# (d)
timer = timeit.Timer(lambda: required_trials(300, 200))
time = timer.timeit(1)
print 'Ran for n = %s, time = %s' % (200, time)

x = []
times = []
m = 300
n = 200
for i in range(0, 40):
	timer = timeit.Timer(lambda: required_trials(m, n))
	time = timer.timeit(1)
	times.append(time)
	x.append(n)
	print 'Ran for n = %s, time = %s' % (n, time)
	n += 500
	m += 120

plt.xlabel('size of range (n)')
plt.ylabel('time in ms (t)')
plt.title('Coupon Collector Running Time')
plt.plot(x, times, 'r')
plt.show()