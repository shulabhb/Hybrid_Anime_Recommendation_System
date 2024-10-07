# collaborative_filtering.py

import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import numpy as np
import time
import joblib
import os
import logging

class CollaborativeFiltering:
    def __init__(self, ratings_df=None, model_path=None, factors=50, iterations=10, regularization=0.01):
        self.logger = logging.getLogger(__name__)
        start_time = time.time()
        if model_path and os.path.exists(model_path):
            self.logger.info("Loading CollaborativeFiltering model from disk...")
            self.load_model(model_path)
            self.logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds.")
        elif ratings_df is not None:
            self.logger.info("Initializing CollaborativeFiltering and training model...")
            # Shuffle user IDs and map to indices
            unique_user_ids = ratings_df['user_id'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_user_ids)
            user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
            ratings_df['user_index'] = ratings_df['user_id'].map(user_id_mapping)

            # Shuffle anime IDs and map to indices
            unique_anime_ids = ratings_df['anime_id'].unique()
            anime_id_mapping = {anime_id: idx for idx, anime_id in enumerate(unique_anime_ids)}
            ratings_df['anime_index'] = ratings_df['anime_id'].map(anime_id_mapping)

            user_indices = ratings_df['user_index'].values
            anime_indices = ratings_df['anime_index'].values
            ratings = ratings_df['rating'].astype(float).values

            # Build mappings between IDs and indices
            self.user_id_to_index = user_id_mapping
            self.index_to_user_id = {idx: user_id for user_id, idx in user_id_mapping.items()}
            self.anime_id_to_index = anime_id_mapping
            self.index_to_anime_id = {idx: anime_id for anime_id, idx in anime_id_mapping.items()}

            # Build the user-item interaction matrix
            self.user_item_data = csr_matrix((ratings, (user_indices, anime_indices)))

            # Initialize ALS model for implicit feedback
            self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)

            # Fit the model
            self.model.fit(self.user_item_data)
            self.logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds.")
        else:
            raise ValueError("Either ratings_df or model_path must be provided.")

    def recommend(self, user_id, k=5):
        start_time = time.time()
        self.logger.info(f"Generating recommendations for user {user_id}...")
        try:
            # Map the user_id to the internal user index
            user_index = self.user_id_to_index.get(user_id)
            if user_index is None:
                self.logger.warning(f"User {user_id} not found in training data.")
                return [], []

            # Get the user's ratings (as a sparse vector)
            user_ratings = self.user_item_data[user_index]

            # Get recommendations, filtering out already liked items
            anime_indices, scores = self.model.recommend(
                user_index,
                user_ratings,
                N=k,
                filter_already_liked_items=True  
            )

            # Map internal anime indices back to anime_ids
            anime_ids = []
            for anime_idx in anime_indices:
                anime_id = self.index_to_anime_id.get(anime_idx)
                if anime_id is None:
                    self.logger.warning(f"Error: anime_idx {anime_idx} not found in index_to_anime_id")
                    continue
                anime_ids.append(anime_id)

            self.logger.info(f"Recommendations generated in {time.time() - start_time:.2f} seconds.")
            return anime_ids, scores

        except Exception as e:
            self.logger.error(f"Error while generating recommendations for user {user_id}: {e}")
            return [], []

    def save_model(self, model_path):
        start_time = time.time()
        self.logger.info(f"Saving CollaborativeFiltering model to {model_path}...")
        model_data = {
            'model': self.model,
            'user_id_to_index': self.user_id_to_index,
            'index_to_user_id': self.index_to_user_id,
            'anime_id_to_index': self.anime_id_to_index,
            'index_to_anime_id': self.index_to_anime_id,
            'user_item_data': self.user_item_data,
        }
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved in {time.time() - start_time:.2f} seconds.")

    def load_model(self, model_path):
        start_time = time.time()
        self.logger.info(f"Loading CollaborativeFiltering model from {model_path}...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.user_id_to_index = model_data['user_id_to_index']
        self.index_to_user_id = model_data['index_to_user_id']
        self.anime_id_to_index = model_data['anime_id_to_index']
        self.index_to_anime_id = model_data['index_to_anime_id']
        self.user_item_data = model_data['user_item_data']
        self.logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds.")
