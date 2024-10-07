# hybrid_recommendation_system.py

import logging
import time

class HybridRecommendationSystem:
    def __init__(self, cf_model, cb_model, alpha=0.5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def get_hybrid_recommendations(self, user_id, anime_df, k=5):
        start_time = time.time()
        self.logger.info(f"Generating hybrid recommendations for user {user_id}...")

        # Check if user exists in CF model
        if user_id not in self.cf_model.user_id_to_index:
            self.logger.warning(f"User {user_id} not found in CF model. Using CB recommendations only.")
            # Provide CB recommendations based on top-rated anime
            top_anime_ids = anime_df.sort_values(by='rating', ascending=False)['anime_id'].tolist()
            # Get top k anime IDs
            recommended_anime_ids = top_anime_ids[:k]
            self.logger.info(f"CB-only recommendations for user {user_id}: {recommended_anime_ids}")
            return recommended_anime_ids
        else:
            # Get CF recommendations
            cf_anime_ids, cf_scores = self.cf_model.recommend(user_id, k=k)
            self.logger.info(f"User {user_id} CF recommendations: {cf_anime_ids}")

            hybrid_scores = {}

            for anime_id, cf_score in zip(cf_anime_ids, cf_scores):
                self.logger.info(f"Processing CF Anime ID {anime_id} with score {cf_score}")
                # Get CB recommendations based on this anime_id
                cb_recommendations = self.cb_model.get_recommendations(anime_id, k=k)
                self.logger.info(f"CB recommendations for Anime ID {anime_id}: {cb_recommendations}")

                # Assign scores to CB recommendations
                for cb_anime_id in cb_recommendations:
                    # Combine scores using alpha
                    if cb_anime_id in hybrid_scores:
                        hybrid_scores[cb_anime_id] += self.alpha * cf_score + (1 - self.alpha)
                    else:
                        hybrid_scores[cb_anime_id] = self.alpha * cf_score + (1 - self.alpha)

            # Sort the hybrid_scores
            sorted_anime_ids = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

            # Get the top k recommendations
            recommended_anime_ids = [anime_id for anime_id, score in sorted_anime_ids[:k]]

            self.logger.info(f"Hybrid recommendations generated in {time.time() - start_time:.2f} seconds.")
            return recommended_anime_ids

    def log_hybrid_recommendations(self, user_id, recommendations, anime_df):
        try:
            # Build a list of recommendations with anime_id and name
            rec_list = []
            for anime_id in recommendations:
                # Get anime name
                anime_name = anime_df[anime_df['anime_id'] == anime_id]['name'].values
                if len(anime_name) > 0:
                    anime_name = anime_name[0]
                else:
                    anime_name = 'Unknown Anime'
                rec_list.append({'user_id': user_id, 'anime_id': anime_id, 'anime_name': anime_name})
            return rec_list
        except Exception as e:
            self.logger.error(f"Error logging recommendations: {e}")
            return []
