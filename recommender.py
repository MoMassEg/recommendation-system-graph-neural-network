"""
Inference helper functions for loading models and generating recommendations.
"""

import os
import torch
import numpy as np
from models import GCNRecommender, GraphSAGERecommender, GATRecommender

class MovieRecommenderSystem:
    def __init__(self, data_path='movielens_graph_data.pt', checkpoint_dir='checkpoints'):
        self.data = torch.load(data_path, weights_only=False)
        self.n_users = self.data['n_users']
        self.n_movies = self.data['n_movies']
        self.genre_dim = self.data['movie_genre_feats'].shape[1]
        self.edge_index = self.data['edge_index']
        self.movie_genre_feats = self.data['movie_genre_feats']
        self.movie_meta = self.data['movie_meta']
        
        # Load models
        self.models = {
            'GCN': GCNRecommender(self.n_users, self.n_movies, self.genre_dim),
            'GraphSAGE': GraphSAGERecommender(self.n_users, self.n_movies, self.genre_dim),
            'GAT': GATRecommender(self.n_users, self.n_movies, self.genre_dim, heads=4)
        }
        
        for name, model in self.models.items():
            ckpt_path = os.path.join(checkpoint_dir, f"{name.lower()}_best.pt")
            if os.path.exists(ckpt_path):
                weights = torch.load(ckpt_path, weights_only=False)
                model.load_state_dict(weights)
            model.eval()
            
        # Build user history lookup from training + val pairs
        self.user_history = {}
        all_pairs = torch.cat([self.data['train_pairs'], self.data['val_pairs'], self.data['test_pairs']], dim=0).numpy()
        all_ratings = torch.cat([self.data['train_ratings'], self.data['val_ratings'], self.data['test_ratings']], dim=0).numpy()
        
        for (u, m), r in zip(all_pairs, all_ratings):
            if u not in self.user_history:
                self.user_history[u] = []
            self.user_history[u].append((m, r))
            
        # Precompute node embeddings for all models for instant recommendation
        self.embeddings = {}
        with torch.no_grad():
            for name, model in self.models.items():
                h_u, h_m = model(self.edge_index, self.movie_genre_feats)
                self.embeddings[name] = (h_u, h_m)
                
    def get_user_history(self, u_idx):
        if u_idx not in self.user_history:
            return []
        # Sort by rating descending
        items = sorted(self.user_history[u_idx], key=lambda x: x[1], reverse=True)
        history = []
        for m_idx, rating in items:
            meta = self.movie_meta.get(m_idx, {})
            history.append({
                'movie_idx': m_idx,
                'title': meta.get('title', f"Movie {m_idx}"),
                'genres': meta.get('genres', []),
                'poster': meta.get('poster', ''),
                'rating': float(rating)
            })
        return history
        
    def recommend_for_user(self, u_idx, model_name='GraphSAGE', top_k=10, exclude_rated=True):
        if model_name not in self.models:
            model_name = 'GraphSAGE'
            
        model = self.models[model_name]
        h_u, h_m = self.embeddings[model_name]
        
        user_emb = h_u[u_idx:u_idx+1].expand(self.n_movies, -1)
        
        with torch.no_grad():
            scores = model.predict_rating(user_emb, h_m).numpy()
            
        rated_movies = set()
        if exclude_rated and u_idx in self.user_history:
            rated_movies = set(m for m, _ in self.user_history[u_idx])
            
        # Sort candidate movies by score
        ranked_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        for m_idx in ranked_indices:
            if m_idx in rated_movies:
                continue
            meta = self.movie_meta.get(int(m_idx), {})
            recommendations.append({
                'movie_idx': int(m_idx),
                'title': meta.get('title', f"Movie {m_idx}"),
                'genres': meta.get('genres', []),
                'poster': meta.get('poster', ''),
                'predicted_rating': float(scores[m_idx])
            })
            if len(recommendations) >= top_k:
                break
                
        return recommendations

    def get_similar_movies(self, target_movie_idx, model_name='GraphSAGE', top_k=5):
        _, h_m = self.embeddings[model_name]
        m_vec = h_m[target_movie_idx:target_movie_idx+1]
        
        # Cosine similarity
        norm_m = torch.norm(m_vec, dim=-1, keepdim=True)
        norm_all = torch.norm(h_m, dim=-1, keepdim=True)
        cos_sim = (torch.mm(m_vec, h_m.t()) / (norm_m * norm_all.t() + 1e-8)).squeeze(0).numpy()
        
        similar_indices = np.argsort(cos_sim)[::-1]
        results = []
        for idx in similar_indices:
            if idx == target_movie_idx:
                continue
            meta = self.movie_meta.get(int(idx), {})
            results.append({
                'movie_idx': int(idx),
                'title': meta.get('title', f"Movie {idx}"),
                'genres': meta.get('genres', []),
                'poster': meta.get('poster', ''),
                'similarity': float(cos_sim[idx])
            })
            if len(results) >= top_k:
                break
        return results

if __name__ == '__main__':
    rec = MovieRecommenderSystem()
    print("Testing recommendations for User 0 using GraphSAGE:")
    recs = rec.recommend_for_user(0, model_name='GraphSAGE', top_k=5)
    for r in recs:
        print(f"  {r['title']} | Score: {r['predicted_rating']:.2f} | Genres: {', '.join(r['genres'])}")
