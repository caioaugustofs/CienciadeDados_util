import matplotlib.pyplot as plt
import numpy as np


class PCA_Metrics:
    """
    A class to represent and visualize PCA metrics.
    Attributes
    ----------
    pca : PCA
        A fitted PCA object from sklearn.
    n : int
        The number of principal components to consider.
    Methods
    -------
    plot_variance():
        Plots the explained variance and cumulative variance of the principal components.
    """
    def __init__(self, pca, n: int=5)-> None:
        self.pca = pca
        self.n = n

    def plot_variance(self) -> None:
        """
        Plots the explained variance and cumulative variance for the PCA components.
        This method creates two subplots:
        1. A bar plot showing the explained variance for each PCA component.
        2. A line plot showing the cumulative explained variance as more components are included.
        The x-axis represents the PCA components, and the y-axis represents the variance.
        Returns:
            None
        """
        evr = self.pca.explained_variance_ratio_[:self.n]
        cv = np.cumsum(evr)
        
        fig, axs = plt.subplots(1, 2)
        grid = np.arange(1, self.n + 1)
        
        axs[0].bar(grid, evr)
        axs[0].set(xlabel='Componentes', title='Variância explicada', ylim=(0.0, 1.0))

        axs[1].plot(np.r_[0, grid], np.r_[0, cv], '-o')
        axs[1].set(xlabel='Componentes', title='Variância acumulada', ylim=(0.0, 1.0))

        fig.suptitle('Análise de PCA', fontsize=16)
        fig.set(figwidth=8)
        fig.tight_layout()
