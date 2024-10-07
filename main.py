# main.py

import time
import pandas as pd
import os
import logging
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based_filtering import ContentBasedFiltering
from models.hybrid_recommendation_system import HybridRecommendationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    if not os.path.exists('output'):
        os.makedirs('output')

    anime_path = 'data/anime.csv'
    rating_path = 'data/rating.csv'
    cf_model_path = 'output/collaborative_filtering_model.pkl'
    data_version_file = 'output/data_version.txt'

    if not os.path.exists(anime_path) or not os.path.exists(rating_path):
        logger.error("Data files not found!")
        return

    try:
        logger.info("Loading data...")
        anime_df = pd.read_csv(anime_path)
        rating_df = pd.read_csv(rating_path)

        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")

        # Handle negative ratings (-1 indicates the user watched but didn't rate)
        rating_df = rating_df[rating_df['rating'] != -1]

        # Ensure data types are consistent
        rating_df['user_id'] = rating_df['user_id'].astype(int)
        rating_df['anime_id'] = rating_df['anime_id'].astype(int)
        rating_df['rating'] = rating_df['rating'].astype(float)

        # Validate and clean datasets
        # Remove duplicate ratings
        duplicate_ratings = rating_df.duplicated().sum()
        logger.info(f"Number of duplicate ratings: {duplicate_ratings}")
        rating_df = rating_df.drop_duplicates()

        # Check for missing values
        missing_values_ratings = rating_df.isnull().sum()
        logger.info(f"Missing values in rating dataset:\n{missing_values_ratings}")
        rating_df = rating_df.dropna()

        missing_values_anime = anime_df.isnull().sum()
        logger.info(f"Missing values in anime dataset:\n{missing_values_anime}")
        anime_df = anime_df.dropna(subset=['anime_id', 'name'])

        # Initialize the Collaborative Filtering model with adjusted parameters
        retrain_model = True  # Force retraining since we've updated the code

        if retrain_model:
            cf_model = CollaborativeFiltering(
                ratings_df=rating_df,
                factors=2,          # Adjusted number of latent factors
                iterations=5,       # Adjusted iterations
                regularization=0.1   # Adjust regularization term
            )
            cf_model.save_model(cf_model_path)
            # Save current data version
            current_data_version = str(os.path.getmtime(rating_path))
            with open(data_version_file, 'w') as f:
                f.write(current_data_version)
        else:
            logger.info("Loading CollaborativeFiltering model from disk...")
            cf_model = CollaborativeFiltering(model_path=cf_model_path)

        # Initialize the Content-Based Filtering model
        cb_model = ContentBasedFiltering(anime_df)

        # Create the Hybrid Recommendation System
        hybrid_model = HybridRecommendationSystem(cf_model, cb_model)

        # Specify the user IDs for whom we want recommendations
        user_ids = [2, 3]

        all_recommendations = []

        for user_id in user_ids:
            logger.info(f"Generating recommendations for user {user_id}...")

            # Check if the user exists in the dataset
            if user_id not in rating_df['user_id'].unique():
                logger.warning(f"User {user_id} not found in the dataset.")
                continue

            # Generate hybrid recommendations
            hybrid_recommendations = hybrid_model.get_hybrid_recommendations(user_id, anime_df, k=5)

            # Map anime IDs to names and collect recommendations
            user_recommendations = hybrid_model.log_hybrid_recommendations(user_id, hybrid_recommendations, anime_df)
            all_recommendations.extend(user_recommendations)

            # Log recommendations to console
            hybrid_anime_names = [rec['anime_name'] for rec in user_recommendations]
            logger.info(f"Hybrid recommendations for user {user_id}: {hybrid_anime_names}")

        # Save all recommendations to one file
        recommendations_df = pd.DataFrame(all_recommendations)
        recommendations_df.to_csv('output/all_recommendations.csv', index=False)
        logger.info(f"All recommendations saved to output/all_recommendations.csv")

    except Exception as e:
        logger.error(f"Error during processing: {e}")

    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
