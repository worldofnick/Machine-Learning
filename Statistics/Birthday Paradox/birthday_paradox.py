import random
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sklearn import preprocessing

def birthdayParadoxTrial(max):
	seen = set()
	iterations = 0
	while True:
		x = random.randint(1, max)
		iterations += 1
		if x in seen:
			break
		seen.add(x)

	return iterations

# (a)
n = 4000
print 'Q1 (a): Trials to get the same value twice %s' % birthdayParadoxTrial(n)

# (b)
m = 300
def birthdayExperiment(m, n):
	required_trials = []
	for i in range(0, m):
		required_trials.append(birthdayParadoxTrial(n))
	return required_trials

required_trials = birthdayExperiment(m, n)
y = [1.0/x for x in required_trials]
data = zip(required_trials, y)
# Choose how many bins you want here
num_bins = m

# Use the histogram function to bin the data
counts, bin_edges = np.histogram(data, bins=num_bins, normed=False)

# Now find the cdf
cdf = np.cumsum(counts)
cdf = preprocessing.MinMaxScaler().fit_transform(cdf)

# And finally plot the cdf
plt.plot(bin_edges[1:], cdf, linewidth=2.0)

plt.xlabel('trials (k)')
plt.ylabel('fraction of success')
plt.title('Birthday Paradox')
plt.show()

# (c)
print 'Q1 (c): Empirical Estimate %f' % (sum(required_trials) / m)

# (d)
timer = timeit.Timer(lambda: birthdayExperiment(300, 4000))
time = timer.timeit(1)
print 'Ran for n = %s, time = %s' % (4000, time)

x = []
times = []
m = 300
n = 4000
for i in range(0, 100):
	timer = timeit.Timer(lambda: birthdayExperiment(m, n))
	time = timer.timeit(1)
	times.append(time)
	x.append(n)
	print 'Ran for n = %s; m = %s, time = %s' % (n, m, time)
	n += 1000
	m += 100
	

plt.xlabel('size of range (n)')
plt.ylabel('time in s (t)')
plt.title('Birthday Paradox Running Time')
plt.plot(x, times, 'b', linewidth=2.0)
plt.show()