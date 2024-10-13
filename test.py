import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('movies.csv')  # Contains movieId, title, genres
ratings = pd.read_csv('ratings.csv')  # Contains userId, movieId, rating
tags = pd.read_csv('tags.csv')  # Contains userId, movieId, tag, timestamp

# Aggregate tags for each movie
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
tags_grouped.columns = ['movieId', 'tags']

# Merge tags with movies
movies_with_tags = pd.merge(movies, tags_grouped, on='movieId', how='left').fillna('')

# Combine genres and tags into a single string
movies_with_tags['combined_features'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tags']


# Create TF-IDF vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_with_tags['combined_features'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    # Find the index of the movie that matches the title
    try:
        idx = movies_with_tags[movies_with_tags['title'].str.contains(title, case=False)].index[0]
    except IndexError:
        return "Movie not found."

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Exclude the first movie (itself)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_with_tags['title'].iloc[movie_indices].tolist()
from sklearn.neighbors import NearestNeighbors

# Create a pivot table for ratings
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Implement Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(ratings_pivot)

def user_based_recommendations(user_id, num_recommendations=10):
    # Get user ratings
    user_ratings = ratings_pivot.loc[user_id].values.reshape(1, -1)
    
    # Get nearest neighbors
    distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=num_recommendations + 1)

    # Get recommended movie IDs
    recommended_movie_indices = indices.flatten()[1:]  # Exclude the first index (self)
    recommended_movie_ids = ratings_pivot.columns[recommended_movie_indices]

    # Fetch movie titles
    recommended_titles = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    
    return recommended_titles

print(user_based_recommendations(1))