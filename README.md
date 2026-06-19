# Hybrid Movie Recommendation System

A movie recommender combining **content-based filtering** and **collaborative filtering**, built on the MovieLens dataset.

## Overview

This project blends two recommendation approaches into a single hybrid model:

- **Content-based filtering**: Uses `CountVectorizer` to convert movie genres into vectors, then computes cosine similarity between movies to recommend titles similar in genre to ones a user already likes.
- **Collaborative filtering**: Uses SVD (Singular Value Decomposition) via the `Surprise` library, trained on ~100,000 user ratings, to predict how a user would rate movies they haven't seen yet.
- **Hybrid blend**: Combines both scores using a tunable `alpha` parameter, allowing the recommendation to lean more heavily on genre similarity or on user rating patterns.

## Results

- Trained and evaluated on the MovieLens dataset (9,742 movies, 100,836 ratings, 610 users)
- Achieved an **RMSE of 0.88** on held-out test data for the collaborative filtering component

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn (CountVectorizer, cosine similarity, MinMaxScaler)
- Surprise (SVD, model evaluation)
- Matplotlib (exploratory data analysis)

## Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) (small version: 100K ratings). Dataset files are not included in this repo — download them directly from the MovieLens website and place `movies.csv` and `ratings.csv` in the project directory to run the notebook.

## Future Improvements

- Enrich content features with cast and director data from TMDB for more nuanced genre-based recommendations
- Experiment with deep learning-based collaborative filtering (e.g. neural matrix factorization)
