import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

def eblow(df, n, title, file_name):
    kmeans_list = [KMeans(n_clusters=k).fit(df.values) for k in range(1, n)]
    sites = [X.cluster_centers_ for X in kmeans_list]
    k_euclid = [cdist(df.values, cent) for cent in sites]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    plt.ylabel('Variance Explained', fontsize=14)
    plt.xlabel('k clusters', fontsize=14)
    plt.title(title, fontsize=15)
    plt.plot(bss)
    fig = plt.gcf()
    plt.show()
    fig.savefig(file_name)


P = pd.read_csv('P.csv')
eblow(P, 10, 'Elbow Method P.csv', 'elbow_P.pdf')
Q = pd.read_csv('Q.csv')
eblow(Q, 12, 'Elbow Method Q.csv', 'elbow_Q.pdf')