"""
Data loader and preprocessor for MovieLens dataset from Hugging Face.
Extracts user-movie interaction graph, genre features, and splits edges for train/val/test.
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_movielens_data(num_users=1200, num_movies=1500, min_ratings=5, random_seed=42):
    print("Loading 'ashraq/movielens_ratings' from Hugging Face...")
    ds = load_dataset("ashraq/movielens_ratings", split="train")
    df = ds.to_pandas()
    
    # Filter to most active users and movies for a dense collaborative graph
    top_users = df['user_id'].value_counts().head(num_users).index
    top_movies = df['movie_id'].value_counts().head(num_movies).index
    
    sub_df = df[df['user_id'].isin(top_users) & df['movie_id'].isin(top_movies)].copy()
    print(f"Filtered interaction count: {len(sub_df)} ratings")
    
    # Create contiguous 0-based indices for users and movies
    unique_users = sorted(sub_df['user_id'].unique())
    unique_movies = sorted(sub_df['movie_id'].unique())
    
    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    movie2idx = {mid: i for i, mid in enumerate(unique_movies)}
    
    sub_df['u_idx'] = sub_df['user_id'].map(user2idx)
    sub_df['m_idx'] = sub_df['movie_id'].map(movie2idx)
    
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    print(f"Unique Users: {n_users}, Unique Movies: {n_movies}")
    
    # Metadata dictionary for movies (title, genres, poster)
    movie_meta = {}
    for _, row in sub_df.drop_duplicates(subset=['m_idx']).iterrows():
        movie_meta[row['m_idx']] = {
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'genres': row['genres'].split('|') if isinstance(row['genres'], str) else [],
            'poster': row.get('posters', '')
        }
    
    # Extract all unique genres to build movie feature vectors
    all_genres = sorted(list(set(g for meta in movie_meta.values() for g in meta['genres'] if g)))
    genre2idx = {g: i for i, g in enumerate(all_genres)}
    print(f"Found {len(all_genres)} unique genres: {all_genres}")
    
    # Multi-hot genre feature for each movie
    movie_genre_feats = np.zeros((n_movies, len(all_genres)), dtype=np.float32)
    for m_idx, meta in movie_meta.items():
        for g in meta['genres']:
            if g in genre2idx:
                movie_genre_feats[m_idx, genre2idx[g]] = 1.0
                
    # Normalize ratings between 0 and 1 or keep as float32
    # Ratings are 0.5 to 5.0
    ratings = sub_df['rating'].values.astype(np.float32)
    
    # In bipartite graph:
    # User nodes: 0 .. n_users - 1
    # Movie nodes: n_users .. n_users + n_movies - 1
    u_nodes = sub_df['u_idx'].values
    m_nodes = sub_df['m_idx'].values + n_users
    
    # Split interactions into Train (80%), Val (10%), Test (10%)
    indices = np.arange(len(sub_df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=random_seed)
    
    # Graph edges (bidirectional for GNN message passing on the training graph)
    train_u = u_nodes[train_idx]
    train_m = m_nodes[train_idx]
    
    # Edge index for training message passing: (u -> m) and (m -> u)
    src = np.concatenate([train_u, train_m])
    dst = np.concatenate([train_m, train_u])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    
    # Supervised rating regression / link prediction targets
    train_pairs = torch.tensor(np.stack([u_nodes[train_idx], m_nodes[train_idx] - n_users], axis=1), dtype=torch.long)
    train_ratings = torch.tensor(ratings[train_idx], dtype=torch.float32)
    
    val_pairs = torch.tensor(np.stack([u_nodes[val_idx], m_nodes[val_idx] - n_users], axis=1), dtype=torch.long)
    val_ratings = torch.tensor(ratings[val_idx], dtype=torch.float32)
    
    test_pairs = torch.tensor(np.stack([u_nodes[test_idx], m_nodes[test_idx] - n_users], axis=1), dtype=torch.long)
    test_ratings = torch.tensor(ratings[test_idx], dtype=torch.float32)
    
    data_dict = {
        'n_users': n_users,
        'n_movies': n_movies,
        'total_nodes': n_users + n_movies,
        'movie_genre_feats': torch.tensor(movie_genre_feats, dtype=torch.float32),
        'edge_index': edge_index,
        'train_pairs': train_pairs,
        'train_ratings': train_ratings,
        'val_pairs': val_pairs,
        'val_ratings': val_ratings,
        'test_pairs': test_pairs,
        'test_ratings': test_ratings,
        'movie_meta': movie_meta,
        'user2idx': user2idx,
        'movie2idx': movie2idx,
        'all_genres': all_genres
    }
    
    print(f"Data prepared successfully! Train: {len(train_ratings)}, Val: {len(val_ratings)}, Test: {len(test_ratings)}")
    return data_dict

if __name__ == '__main__':
    data = prepare_movielens_data()
    torch.save(data, 'movielens_graph_data.pt')
    print("Saved 'movielens_graph_data.pt'.")
