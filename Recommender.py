import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


file_path = '../Spotify_music_dataset/spotify_data.csv'
music_data = pd.read_csv(file_path)


columns_to_drop = ['Unnamed: 0', 'track_id', 'year']
music_data = music_data.drop(columns_to_drop, axis=1)


# Replacing the missing numerical values with the mean of each - to prevent outliers in the data
numerical_columns = music_data.select_dtypes(include=['float64', 'int64']).columns
music_data[numerical_columns] = music_data[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
# Fill missing values for 'artist_name' and 'track_name' with a placeholder 
music_data[['artist_name', 'track_name']] = music_data[['artist_name', 'track_name']].fillna('Unknown')


train_set, test_set = train_test_split(music_data, test_size=0.2, random_state=42)
music_data = train_set.copy()


# Step 1: Preprocess the data
def preprocess_data(data):
    feature_columns = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity'
    ]
    # Normalize the features
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data, feature_columns

# Step 2: Apply K-means clustering
def apply_kmeans(data, feature_columns, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[feature_columns])
    return kmeans, data





# Step 3: Recommendation system combining K-means and cosine similarity
def recommend_songs_combined(song_name, data, feature_columns, kmeans, top_n=5):
    # Find the song index by name
    song_index = data[data['track_name'].str.lower() == song_name.lower()].index
    
    if len(song_index) == 0:
        return "Song not found in the dataset."
    
    # Get the first index (in case there are duplicates)
    index = song_index[0]

    # Get the cluster of the target song
    target_cluster = data.loc[index, 'cluster']

    # Filter data to only include songs in the same cluster
    cluster_data = data[data['cluster'] == target_cluster]

    # Get feature vector of the target song
    target_features = data.loc[index, feature_columns].values.reshape(1, -1)

    # Compute cosine similarity within the same cluster
    cluster_features = cluster_data[feature_columns].values
    similarity_scores = cosine_similarity(target_features, cluster_features)[0]

    # Add similarity scores to the cluster data
    cluster_data = cluster_data.copy()
    cluster_data['similarity_score'] = similarity_scores

    # Exclude the target song from recommendations
    cluster_data = cluster_data[cluster_data.index != index]

    # Sort by similarity and popularity
    recommendations = cluster_data.sort_values(
        by=['similarity_score', 'popularity'], ascending=[False, False]
    ).head(top_n)

    # Get input song information
    input_song_info = data.loc[index, ['track_name', 'artist_name', 'popularity', 'cluster'] + feature_columns].to_dict()

    return input_song_info, recommendations[['track_name', 'artist_name', 'popularity', 'similarity_score', 'cluster']]

# Preprocess the dataset
music_data, feature_columns = preprocess_data(music_data)

# Apply K-means clustering
kmeans_model, music_data = apply_kmeans(music_data, feature_columns, n_clusters=10)

# Example usage
song_name = 'shape of you'  # Replace with your desired song
input_song_info, recommendations = recommend_songs_combined(song_name, music_data, feature_columns, kmeans_model)

print("Input Song Information:")
print(pd.DataFrame([input_song_info])) # Display the input song details


print("\nRecommended Songs:")
print(recommendations)  # Display the recommended songs
