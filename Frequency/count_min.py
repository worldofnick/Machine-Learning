import sys

def hash(x, p, m):
	x = x % p
	return x % m

primes = [5, 7, 11, 13, 17]

def count_min(stream, k):

	m = len(stream)

	counters = [[0]*k for i in range(0, len(primes))]
	
	for item in stream:
		for i in range(0, len(primes)):
			counters[i][hash(ord(item), primes[i], m) % k] += 1

	return counters

def estimate(item, counters, m, k):
	min_value = sys.maxint

	for i in range(0, len(primes)):
		min_value = min(min_value, counters[i][hash(ord(item), primes[i], m) % k])
	return min_value

s1_text = open("/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw4/S2.txt", "r")
lines = s1_text.read()
s1 = list(lines)
k = 10
counters = count_min(s1, k)

print counters
print estimate('a', counters, len(s1), k)
print estimate('b', counters, len(s1), k)
print estimate('c', counters, len(s1), k)

