import scipy as sp
import numpy as np
from scipy import linalg as LA
from sklearn.preprocessing import scale
from sklearn import decomposition

A = np.loadtxt(open("A.csv"),delimiter=",",skiprows=0)
U, s, Vt = LA.svd(A, full_matrices=False)

# (a) the second right singular vector
print 'The second right singular vector is: %s' % (Vt[1])

# (b) the fourth singular value
print 'The fourth singular value is: %s' % (s[3])

# (c) the third left singular vector
print 'The third left singular vector is: %s' %(U[:,2])

# the number of non-zero singular values reports the rank
# (d) What is the rank of A?
rank = 0
for i in range(len(s)):
	if s[i] != 0:
		rank = rank + 1

print 'The rank of A is %s' % (rank)

print s

# (e), starting point is 4, d = 4, thereore just sum the last element and square
print '(e) %s' % (s[3]**2)

# (f)
print '(f) %s' % (s[3]**2)

# Center the matrix before PCA
centerA = scale(A)
U, s, Vt = LA.svd(centerA, full_matrices=False)

# (g), starting point is 4, d = 4, thereore just sum the last element and square
print '(g) %s' % (s[3]**2)

# (h)
print '(h) %s' % (s[3]**2)

