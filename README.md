# Anime Recommendation System

## Overview

The **Anime Recommendation System** is a machine learning-based project designed to recommend anime to users based on both collaborative filtering and content-based filtering.
The system preprocesses and cleans anime metadata and user data, then applies vectorization techniques to generate genre vectors for each anime. Using the processed data, the system provides personalized anime recommendations.

This is the first stage of our model. We now begin the testing and tuning of this model.

## Features
- **Data Cleaning & Preprocessing**: Handles missing and malformed values in anime metadata and user data.
- **Genre Vectorization**: Uses scikit-learn’s CountVectorizer to generate feature vectors based on anime genres.
- **User Data Integration**: Merges normalized user data (ratings, preferences) with anime data for improved recommendations.
- **Recommendation Algorithms**: Implements both content-based and collaborative filtering algorithms to provide recommendations.

## Technologies
- Python (pandas, NumPy)
- scikit-learn (CountVectorizer, MinMaxScaler)
- Machine Learning Techniques
- Data Preprocessing

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AnimeRecommendationSystem.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the data preprocessing script:
   ```bash
   python data_cleaning_and_preprocessing.py
   ```
4. Implement the recommendation algorithm in `recommendation_system.py` (example coming soon).

## Future Work
- Implement advanced recommendation algorithms such as matrix factorization.
- Expand dataset to include user reviews and tags for better content-based recommendations.
- Improve UI/UX for end-user interaction with the system.


---
