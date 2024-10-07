# content_based_filtering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import time
import logging

class ContentBasedFiltering:
    def __init__(self, anime_df):
        self.logger = logging.getLogger(__name__)
        start_time = time.time()
        self.logger.info("Initializing ContentBasedFiltering...")
        self.anime_df = anime_df.reset_index(drop=True)
        self.anime_df['genre'] = self.anime_df['genre'].fillna('')
        self.anime_df['type'] = self.anime_df['type'].fillna('')
        self.anime_df['episodes'] = self.anime_df['episodes'].astype(str).fillna('')

        # Since 'synopsis' is not available, we'll use 'genre', 'type', and 'episodes' as content
        self.anime_df['content'] = self.anime_df['genre'] + ' ' + self.anime_df['type'] + ' ' + self.anime_df['episodes']

        # Create a mapping from anime_id to DataFrame index
        self.anime_id_to_index = pd.Series(self.anime_df.index, index=self.anime_df['anime_id']).to_dict()
        self.index_to_anime_id = pd.Series(self.anime_df['anime_id'], index=self.anime_df.index).to_dict()

        # Compute the TF-IDF matrix on the content
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.anime_df['content'])

        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        self.logger.info(f"Content-based model initialized in {time.time() - start_time:.2f} seconds.")

    def get_recommendations(self, anime_id, k=5):
        start_time = time.time()
        self.logger.info(f"Generating content-based recommendations for anime ID {anime_id}...")
        try:
            # Get the index of the given anime_id
            idx = self.anime_id_to_index.get(anime_id)
            if idx is None:
                self.logger.warning(f"Anime ID {anime_id} not found in anime_id_to_index mapping.")
                return []

            # Get pairwise similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Sort the anime based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the indices of the most similar anime (excluding itself)
            sim_scores = sim_scores[1:k+1]
            anime_indices = [i[0] for i in sim_scores]

            # Get the recommended anime IDs
            recommendations = [self.anime_df.iloc[i]['anime_id'] for i in anime_indices]

            self.logger.info(f"Recommendations generated in {time.time() - start_time:.2f} seconds.")
            return recommendations

        except Exception as e:
            self.logger.error(f"Error while generating recommendations for anime {anime_id}: {e}")
            return []
