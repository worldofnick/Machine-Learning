import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [(-3000,0), (1,2), (25573,4), (2034,10), (1300, -100)]

plt.plot([-3000,1,25573,2034,1300], [0,2,4,10,-100], 'ro')
plt.title('Bad data set for Lloyds algorithm')
plt.xlabel('x')
plt.ylabel('y')
fig = plt.gcf()
#plt.show()
fig.savefig('bad_data.pdf')

for i in range(0, 20):
	kmeans = KMeans(init='random', n_clusters=3, random_state=i, max_iter=4000, n_init=1)
	kmeans.fit(X)
	print kmeans.inertia_
	print kmeans.cluster_centers_
