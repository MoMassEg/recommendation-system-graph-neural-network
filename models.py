"""
GNN Architectures for Movie Recommendation:
1. GCN (Graph Convolutional Network) - Spectral normalized neighbor aggregation
2. GraphSAGE (Sample and Aggregate) - Inductive neighbor aggregation with self-concatenation
3. GAT (Graph Attention Network) - Attention-weighted neighbor aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class BaseGNNRecommender(nn.Module):
    def __init__(self, n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim
        
        # Node embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # Genre feature projection
        self.genre_proj = nn.Linear(genre_dim, embedding_dim)
        
        # Initialize embeddings with small normal distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        # Link / Rating Predictor MLP
        # Takes [h_u, h_m, h_u * h_m]
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def get_initial_node_features(self, movie_genres):
        # User initial features
        u_feat = self.user_embedding.weight # (n_users, dim)
        
        # Movie initial features = movie_embedding + genre projection
        m_feat = self.movie_embedding.weight + self.genre_proj(movie_genres) # (n_movies, dim)
        
        # Concatenate into full graph node features: [Users; Movies]
        x = torch.cat([u_feat, m_feat], dim=0) # (n_users + n_movies, dim)
        return x
        
    def predict_rating(self, h_u, h_m):
        # Element-wise product provides symmetric similarity signal
        interaction = torch.cat([h_u, h_m, h_u * h_m], dim=-1)
        pred = self.predictor(interaction).squeeze(-1)
        # Constrain rating prediction to reasonable bounds [0.5, 5.0]
        pred = torch.sigmoid(pred) * 4.5 + 0.5
        return pred

class GCNRecommender(BaseGNNRecommender):
    def __init__(self, n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__(n_users, n_movies, genre_dim, embedding_dim, hidden_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_index, movie_genres):
        x = self.get_initial_node_features(movie_genres)
        
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Split into user and movie embeddings
        h_users = h[:self.n_users]
        h_movies = h[self.n_users:]
        return h_users, h_movies

class GraphSAGERecommender(BaseGNNRecommender):
    def __init__(self, n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__(n_users, n_movies, genre_dim, embedding_dim, hidden_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_index, movie_genres):
        x = self.get_initial_node_features(movie_genres)
        
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        h_users = h[:self.n_users]
        h_movies = h[self.n_users:]
        return h_users, h_movies

class GATRecommender(BaseGNNRecommender):
    def __init__(self, n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__(n_users, n_movies, genre_dim, embedding_dim, hidden_dim)
        # Multi-head attention: heads * (hidden_dim // heads) = hidden_dim
        head_dim = hidden_dim // heads
        self.conv1 = GATConv(embedding_dim, head_dim, heads=heads, dropout=dropout)
        # Second layer with 1 head producing hidden_dim
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_index, movie_genres):
        x = self.get_initial_node_features(movie_genres)
        
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        
        h_users = h[:self.n_users]
        h_movies = h[self.n_users:]
        return h_users, h_movies
