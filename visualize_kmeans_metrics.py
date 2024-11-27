import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from typing import List, Tuple, Any

class Visualize_kmeans_metrics:
    """
    Esses métodos ajudam a visualizar a qualidade do clustering para diferentes
    números de clusters, facilitando a escolha do número ideal 
    de clusters para os dados.
    """

    
    def __init__(self,
                 data: Any = None,
                 range_cluster: Tuple[int, int] = (2, 10),
                 seed: int = 42,
                 max_iter: int = 100,
                 n_init: int = 10):
        
        """
        Initialize the KMeans visualization class.
        Parameters:
        data (any): The dataset to be used for clustering. Default is None.
        range_cluster (tuple): A tuple specifying the range of clusters to evaluate. Default is (2, 10).
        seed (int): The random seed for reproducibility. Default is 42.
        max_iter (int): The maximum number of iterations for the KMeans algorithm. Default is 100.
        n_init (int): The number of time the k-means algorithm will be run with different centroid seeds. Default is 10.
        """
        
        self.data = data
        self.range_cluster = range_cluster
        self.seed = seed
        self.max_iter = max_iter
        self.n_init = n_init

    def KMeans_metrics(self) -> Tuple[List[float], List[float]]:
        """
        Calculate KMeans clustering metrics for a range of cluster numbers.
        This method computes the distortions (inertia) and silhouette scores for 
        KMeans clustering over a specified range of cluster numbers. The results 
        are used to evaluate the quality of clustering.
        Returns:
            tuple[list[float], list[float]]: A tuple containing two lists:
                - distortions: List of distortion values (inertia) for each number of clusters.
                - sil_score: List of silhouette scores for each number of clusters.
        """

        sil_score: List[float] = []
        distortions: List[float] = []

        range_int, range_end = self.range_cluster

        for i in range(range_int, range_end):

            kmeans = KMeans(n_clusters=i,
                            n_init=self.n_init,
                            init="k-means++",
                            max_iter=self.max_iter,
                            random_state=self.seed)
            
            kmeans.fit(self.data)
            labels = kmeans.labels_
            distortions.append(kmeans.inertia_)
            sil_score.append(silhouette_score(self.data, labels))

        return distortions, sil_score
    

    def plt_distortions(self, distortions: List[float] = None, figsize: Tuple[int, int] = (10, 4)) -> None:
        """
        Plots the distortions for different numbers of clusters.
        Parameters:
        distortions (list[float], optional): A list of distortion values for different numbers of clusters. 
                                             If None, a ValueError is raised. Default is None.
        figsize (tuple[int, int], optional): A tuple representing the size of the figure. Default is (10, 4).
        Raises:
        ValueError: If distortions is None.
        Returns:
        None
        """
        self.distortions = distortions
        if distortions is None:
            raise ValueError("distortions is None")
        
        range_int, range_end = self.range_cluster
     
        plt.figure(figsize=figsize)
        plt.plot(range(range_int, range_end, 1), distortions, alpha=0.5)
        plt.plot(range(range_int, range_end, 1), distortions, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Distortion")
        plt.show()

    def plt_sil_score(self, sil_score: List[float] = None, figsize: Tuple[int, int] = (10, 4)) -> None:
        """
        Plots the silhouette score for different numbers of clusters.
        Parameters:
        sil_score (list[float], optional): A list of silhouette scores for different numbers of clusters. 
                                           If None, a ValueError is raised. Defaults to None.
        figsize (tuple[int, int], optional): A tuple representing the size of the figure. Defaults to (10, 4).
        Raises:
        ValueError: If sil_score is None.
        Returns:
        None
        """
        self.sil_score = sil_score
        if sil_score is None:
            raise ValueError("sil_score is None")
        
        range_int, range_end = self.range_cluster

        plt.figure(figsize=figsize)
        plt.plot(range(range_int, range_end, 1), sil_score)
        plt.title("Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()

