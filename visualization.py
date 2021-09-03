import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
import numpy as np
from dataio import *
from sklearn.decomposition import PCA

matplotlib.rcParams.update({
    'image.cmap': 'inferno',
})

longnames = {
        'cnv': 'Copy number variation',
        'crp': 'CRIPR Cas9 knockout',
        'exp': 'Gene expression',
        'prot': 'Proteomics',
        }

def allfeatures(show=False):
    d = np.concatenate(
            (read("cnv.pickle"),
            read("crp.pickle"),
            read("exp.pickle"),
            read("prot.pickle")),
            axis=1).T
    pca = PCA(n_components=2, random_state=0).fit(d)
    d = pca.transform(d)
    y = d[:, 0]
    x = d[:, 1]
    plt.scatter(x, y, s=5, c='black')
    if show: plt.show()
    else: plt.close()

def clusters(show=False, centroids=False):
    n_clusters = 7
    clusterings = read('clusterings.pickle')
    for c in clusterings:
        clustering = clusterings[c][n_clusters-1]
        nearestidx = clustering[1]
        colors = clustering[0].labels_
        d = read(f"{c}.pickle").T
        pca = PCA(n_components=2, random_state=0).fit(d)
        dpca = pca.transform(d)
        npca = dpca[nearestidx]
        cpca = pca.transform(clustering[0].cluster_centers_)
        plt.scatter(dpca[:,0], dpca[:,1], s=1, c=colors, vmin=-1, vmax=n_clusters)
        if centroids:
            plt.scatter(npca[:,0], npca[:,1], marker='+', c='black', label='Feature closest to centroid')
            plt.scatter(cpca[:,0], cpca[:,1], marker='x', c='black', label='Centroid')
            plt.legend(framealpha=1, fancybox=False, loc='lower right')
            savefig(f"figures/{c}_clustering_centroids")
        else:
            savefig(f"figures/{c}_clustering")
        if show: plt.show()
        else: plt.close()

def scores(show=False, sub=False):
    scores = read('scores.pickle')
    colors = pl.cm.inferno(np.linspace(0.1, 0.9, len(scores)))
    colidx = 0
    for s in scores:
        y = scores[s]
        if sub: y = y[:20]
        x = range(1, len(y)+1)
        if sub: plt.xticks(x)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.plot(x, y, label=longnames[s], color=colors[colidx])
        colidx += 1
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    plt.legend(fancybox=False)
    if sub:
        savefig("figures/scores_sub")
    else:
        savefig("figures/scores")
    if show: plt.show()
    else: plt.close()

def correlations():
    correlations = read('correlations.pickle')
    print('  cnv', ' crp', ' exp', ' prot')
    print(np.round(correlations, 2))

def nullsums():
    nullsums = read('nullsums.pickle')
    plt.plot(range(len(nullsums)), nullsums)
    plt.xlabel('Drug')
    plt.ylabel('N of nulls')
    if show: plt.show()

def shapes():
    for i in longnames:
        t = read(f"{i}.pickle")
        print(i)
        print(t.shape)


#correlations()
#nullsums()
#shapes()

clusters(centroids=True)
scores()
