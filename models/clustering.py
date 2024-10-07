# clustering.py

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

class OptimizedClustering:
    def __init__(self, ratings_df, n_components=50):
        self.logger = logging.getLogger(__name__)
        start_time = time.time()
        self.logger.info("Initializing OptimizedClustering...")
        # Build the user-item interaction matrix as a sparse matrix
        user_id_categories = ratings_df['user_id'].astype('category')
        anime_id_categories = ratings_df['anime_id'].astype('category')
        user_codes = user_id_categories.cat.codes
        anime_codes = anime_id_categories.cat.codes
        ratings = ratings_df['rating']

        # Build the user-item interaction matrix
        self.user_item_data = csr_matrix((ratings, (user_codes, anime_codes)))

        # Apply dimensionality reduction
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.ratings_reduced = self.svd.fit_transform(self.user_item_data)

        self.logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds.")

    def find_optimal_clusters(self, max_clusters=20):
        start_time = time.time()
        self.logger.info("Finding optimal clusters...")
        distortions = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
            kmeans.fit(self.ratings_reduced)
            distortions.append(kmeans.inertia_)

            silhouette_avg = silhouette_score(self.ratings_reduced, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            self.logger.info(f"Cluster: {k}, Silhouette Score: {silhouette_avg}")

        # Plotting the Elbow Method graph and Silhouette Score graph
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), distortions, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')

        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')

        plt.tight_layout()
        plt.show()
        self.logger.info(f"Optimal cluster finding completed in {time.time() - start_time:.2f} seconds.")

    def cluster_users(self, n_clusters=10):
        start_time = time.time()
        self.logger.info(f"Clustering users into {n_clusters} clusters...")
        try:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            self.user_clusters = kmeans.fit_predict(self.ratings_reduced)
            self.logger.info(f"User clusters: {self.user_clusters}")
        except Exception as e:
            self.logger.error(f"Error during user clustering: {e}")
        self.logger.info(f"User clustering completed in {time.time() - start_time:.2f} seconds.")

    def save_clusters(self, filepath='output/user_clusters.npy'):
        """
        Save the user clusters to a file.
        """
        try:
            if not os.path.exists('output'):
                os.makedirs('output')
            np.save(filepath, self.user_clusters)
            self.logger.info(f"User clusters saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving clusters: {e}")

# Usage Example
def main():
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    anime_path = 'data/anime.csv'
    rating_path = 'data/rating.csv'

    if not os.path.exists(anime_path) or not os.path.exists(rating_path):
        logger.error("Data files not found!")
        return

    # Load the data
    logger.info("Loading data...")
    anime_df = pd.read_csv(anime_path)
    rating_df = pd.read_csv(rating_path)
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")

    # Initialize the optimized clustering
    clustering = OptimizedClustering(rating_df)

    # Find the optimal number of clusters
    clustering.find_optimal_clusters(max_clusters=10)

    # Perform user clustering with the optimal number of clusters
    clustering.cluster_users(n_clusters=5)

    # Save the clusters
    clustering.save_clusters()
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
