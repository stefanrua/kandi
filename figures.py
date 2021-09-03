import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dataio import *
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

matplotlib.rcParams.update({
    'image.cmap': 'inferno',
})

def anscombe():
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    datasets = {
        'I': (x, y1),
        'II': (x, y2),
        'III': (x, y3),
        'IV': (x4, y4)
    }

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6),
                            gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
    axs[0, 0].set(xlim=(0, 20), ylim=(2, 14))
    axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12))

    for ax, (label, (x, y)) in zip(axs.flat, datasets.items()):
        ax.tick_params(direction='in', top=True, right=True)

        # linear regression
        p1, p0 = np.polyfit(x, y, deg=1)  # slope, intercept
        ax.axline(xy1=(0, p0), slope=p1, color='darkgray', lw=2)

        # scatter
        ax.plot(x, y, 'o', color='black')

    savefig('figures/anscombe')
    plt.show()

def kmeans_assumptions():
    def savefig(fname):
        plt.savefig(f"{fname}.pdf")
        plt.savefig(f"{fname}.png", dpi=300)
        print('Wrote', fname)

    vmx=2.5
    vmn=-1
    s=2
    fontsize=10

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, vmin=vmn, vmax=vmx, s=s)
    plt.title("Incorrect Number of Blobs", fontsize=fontsize)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    # Anisotropically distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred, vmin=vmn, vmax=vmx, s=s)
    plt.title("Anisotropically Distributed Blobs", fontsize=fontsize)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

    plt.subplot(223)
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred, vmin=vmn, vmax=vmx, s=s)
    plt.title("Unequal Variance", fontsize=fontsize)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
                    random_state=random_state).fit_predict(X_filtered)

    plt.subplot(224)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred, vmin=vmn, vmax=vmx, s=s)
    plt.title("Unevenly Sized Blobs", fontsize=fontsize)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    savefig('figures/kmeans_assumptions')
    plt.show()

def kpca():
    fontsize=10

    np.random.seed(0)

    X, y = make_circles(n_samples=400, factor=.3, noise=.05)

    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Plot results

    plt.figure()
    plt.subplot(2, 2, 1, aspect='equal')
    plt.title("Original space", fontsize=fontsize)
    reds = y == 0
    blues = y == 1

    plt.scatter(X[reds, 0], X[reds, 1], c="orange",
                s=20)
    plt.scatter(X[blues, 0], X[blues, 1], c="tab:red",
                s=20)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    # projection on the first principal component (in the phi space)
    Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

    plt.subplot(2, 2, 2, aspect='equal')
    plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="orange",
                s=20)
    plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="tab:red",
                s=20)
    plt.title("Projection by PCA", fontsize=fontsize)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.subplot(2, 2, 3, aspect='equal')
    plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="orange",
                s=20)
    plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="tab:red",
                s=20)
    plt.title("Projection by KPCA", fontsize=fontsize)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.subplot(2, 2, 4, aspect='equal')
    plt.scatter(X_back[reds, 0], X_back[reds, 1], c="orange",
                s=20)
    plt.scatter(X_back[blues, 0], X_back[blues, 1], c="tab:red",
                s=20)
    plt.title("Original space after inverse transform", fontsize=fontsize)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.tight_layout()
    savefig('figures/kpca')
    plt.show()

anscombe()
kmeans_assumptions()
kpca()
