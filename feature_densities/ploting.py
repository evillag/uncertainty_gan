import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def mean_likelihood(likelihoods):
    return likelihoods.mean(1)


def log_likelihood(likelihoods):
    likelihoods = likelihoods.copy()
    likelihoods[likelihoods == 0] = .01
    return np.log(likelihoods).sum(1)


# Plot functions
def plot_hist_kde(ax, d, data, kde):
    x = np.linspace(data.min(), data.max(), 100)
    ax.set_title(f'Dimension {d + 1}')
    ax.hist(data, 25, density=True)
    ax.plot(x, kde(x))


def estimate_likelihoods(train_data, test_data, eps=.01, n_samples=None, n_features=None, plot_shape=None):
    if not n_samples:
        n_samples = test_data.shape[0]  # 524521 observations

    if not n_features:
        n_features = train_data.shape[1]  # 128 dimensions

    if plot_shape:
        n_subplots = plot_shape[0] * plot_shape[1]

        if n_features != n_subplots:
            raise ValueError(f'Cannot plot {n_features} features in {n_subplots} subplots.')

        rows, columns = plot_shape
        fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))

    likelihoods = np.zeros((n_samples, n_features))

    for j in range(n_features):
        train_sample = train_data[:10000, j]
        kde = gaussian_kde(train_sample)

        for i in range(n_samples):
            point = test_data[i, j]
            likelihoods[i, j] = kde.integrate_box_1d(point - eps, point + eps)

        if plot_shape:
            plot_hist_kde(axes.flat[j], j, train_sample, kde)

    if plot_shape:
        plt.tight_layout()
        plt.show()

    return likelihoods


def plot_likelihoods(likelihoods):
    plt.title('Likelihoods')
    plt.imshow(likelihoods)
    plt.xlabel('Features')
    plt.ylabel('Observations')
    plt.colorbar()
    plt.show()


def plot_likelihoods(ax, x, likelihoods, label):
    ax.bar(x, likelihoods, 1, label=label)
    ax.legend()


def plot_ue(likelihoods):
    log_likelihoods = log_likelihood(likelihoods)
    mean_likelihoods = mean_likelihood(likelihoods)

    fig, axes = plt.subplots(2, 1, sharex=True)
    x = range(likelihoods.shape[0])

    axes[0].set_title('Uncertainty scores')
    plot_likelihoods(axes[0], x, -log_likelihoods, 'Negative log likelihoods')
    plot_likelihoods(axes[1], x, mean_likelihoods, 'Mean likelihoods')
    axes[1].set_xlabel('Observations')
    plt.show()
