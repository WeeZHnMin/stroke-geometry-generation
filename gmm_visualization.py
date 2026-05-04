import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def gaussian_2d(pos, mean, cov):
    """Compute the 2D Gaussian density at every point in pos."""
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    diff = pos - mean
    exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    return norm * np.exp(-0.5 * exponent)


def gaussian_mixture_2d(pos, weights, means, covariances):
    density = np.zeros(pos.shape[:-1])
    for weight, mean, cov in zip(weights, means, covariances):
        density += weight * gaussian_2d(pos, mean, cov)
    return density


def main():
    # Three Gaussian components. You can change these parameters.
    weights = np.array([0.45, 0.35, 0.20])
    means = np.array([
        [-2.0, -1.0],
        [1.5, 1.0],
        [0.0, -2.5],
    ])
    covariances = np.array([
        [[1.0, 0.3], [0.3, 0.7]],
        [[0.8, -0.2], [-0.2, 1.2]],
        [[0.5, 0.0], [0.0, 0.4]],
    ])

    x = np.linspace(-5, 5, 220)
    y = np.linspace(-5, 5, 220)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = gaussian_mixture_2d(pos, weights, means, covariances)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    heatmap = ax1.imshow(
        Z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax1.contour(X, Y, Z, levels=12, colors="white", linewidths=0.7, alpha=0.8)
    ax1.scatter(means[:, 0], means[:, 1], c="red", s=60, marker="x", label="means")
    ax1.set_title("Gaussian Mixture Density: Heatmap + Contours")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    fig.colorbar(heatmap, ax=ax1, label="p(x, y)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_title("Gaussian Mixture Density: 3D Surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("p(x, y)")
    ax2.view_init(elev=28, azim=-55)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
