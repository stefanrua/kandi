import numpy as np
import time
from dataio import *
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

matrix = read('matrix.pickle')
cnv    = read('cnv.pickle')
crp    = read('crp.pickle')
exp    = read('exp.pickle')
prot   = read('prot.pickle')

tables = [matrix, cnv, crp, exp, prot]
tablenames = ['matrix', 'cnv', 'crp', 'exp', 'prot']
ntables = len(tables)

max_clust = 100

def clustering():
    print('clustering')
    clusterings = {
            'cnv': [],
            'crp': [],
            'exp': [],
            'prot': [],
            }
    for i in range(1, ntables):
        d = tables[i].T
        c = []
        for j in range(max_clust):
            print(f"n clusters = {j+1}   ", end='')
            start = time.time()
            clustering = KMeans(n_clusters=j+1, random_state=0).fit(d)
            nearestidx, _ = pairwise_distances_argmin_min(clustering.cluster_centers_, d)
            c.append([clustering, nearestidx])
            diff = time.time() - start
            print(f"{diff:.3f} s")
        clusterings[tablenames[i]] = c
    write(clusterings, 'clusterings.pickle')

def pca():
    print('pca')
    pcas = {
            'cnv': [],
            'crp': [],
            'exp': [],
            'prot': [],
            }
    for i in range(1, ntables):
        d = tables[i]
        print('linear')
        pca = PCA(n_components=max_clust, random_state=0).fit_transform(d)
        print('rbf')
        pca_rbf = KernelPCA(n_components=max_clust, kernel='rbf', random_state=0).fit_transform(d)
        pcas[tablenames[i]] = [pca, pca_rbf]
    write(pcas, 'pcas.pickle')


def regression():
    print('regression')
    clusterings = read('clusterings.pickle')
    scores = {
            'cnv': [],
            'crp': [],
            'exp': [],
            'prot': [],
            }
    for i in range(1, ntables):
        X_train, X_test, y_train, y_test = train_test_split(\
                tables[i], matrix, test_size=0.33, random_state=0)
        for j in range(max_clust):
            idx = clusterings[tablenames[i]][j][1]
            X_train_clust = X_train[:, idx]
            X_test_clust = X_test[:, idx]
            reg=Ridge().fit(X_train_clust, y_train)
            scores[tablenames[i]].append(reg.score(X_test_clust, y_test))
    write(scores, 'scores.pickle')

def correlation():
    print('correlation')
    ndrugs = matrix.shape[1]
    correlations = np.empty([ndrugs, ntables-1])
    for i in range(ndrugs):
        print(f"drug {i+1}/{ndrugs}")
        for j in range(ntables-1):
            t = tables[j+1]
            corhigh = 0
            for k in range(t.shape[1]):
                drug = matrix[:,i].astype(float)
                feature = t[:,k].astype(float)
                cor = np.corrcoef(drug, feature)[0, 1]
                corhigh = max(abs(corhigh), abs(cor))
            correlations[i][j] = corhigh
    write(correlations, 'correlations.pickle')

clustering()
regression()
#correlation()
#pca()
