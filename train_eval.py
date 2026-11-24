"""
Training and Evaluation Pipeline for GCN, GraphSAGE, and GAT.
Computes:
- Rating Metrics: RMSE, MAE
- Ranking Metrics: Precision@10, Recall@10, NDCG@10, HitRate@10
- Training Runtime & Convergence logs
Saves best checkpoints, metrics JSON, and training curve plots.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models import GCNRecommender, GraphSAGERecommender, GATRecommender

def evaluate_metrics(model, edge_index, movie_genres, test_pairs, test_ratings, n_users, n_movies, k=10, relevance_thresh=3.5):
    model.eval()
    with torch.no_grad():
        h_u, h_m = model(edge_index, movie_genres)
        
        # 1. Rating Prediction (RMSE, MAE)
        u_idx = test_pairs[:, 0]
        m_idx = test_pairs[:, 1]
        
        # Predict ratings in chunks to conserve memory
        preds = []
        chunk_size = 10000
        for i in range(0, len(test_pairs), chunk_size):
            chunk_u = u_idx[i:i+chunk_size]
            chunk_m = m_idx[i:i+chunk_size]
            pred = model.predict_rating(h_u[chunk_u], h_m[chunk_m])
            preds.append(pred)
        preds = torch.cat(preds, dim=0)
        
        mse = torch.mean((preds - test_ratings) ** 2).item()
        rmse = float(np.sqrt(mse))
        mae = float(torch.mean(torch.abs(preds - test_ratings)).item())
        
        # 2. Ranking Metrics: Precision@K, Recall@K, NDCG@K, HitRate@K
        # Organize test pairs by user
        user_test_dict = {}
        for (u, m), r, p in zip(test_pairs.cpu().numpy(), test_ratings.cpu().numpy(), preds.cpu().numpy()):
            if u not in user_test_dict:
                user_test_dict[u] = {'movies': [], 'ratings': [], 'preds': []}
            user_test_dict[u]['movies'].append(m)
            user_test_dict[u]['ratings'].append(r)
            user_test_dict[u]['preds'].append(p)
            
        precisions, recalls, ndcgs, hit_rates = [], [], [], []
        
        for u, val in user_test_dict.items():
            movies = np.array(val['movies'])
            ratings = np.array(val['ratings'])
            scores = np.array(val['preds'])
            
            # Ground truth relevant movies for user in test set
            relevant_mask = ratings >= relevance_thresh
            num_relevant = np.sum(relevant_mask)
            if num_relevant == 0:
                continue
                
            # Rank test items by predicted score
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_relevant = relevant_mask[top_k_indices]
            
            # Precision@K
            p_k = np.sum(top_k_relevant) / k
            # Recall@K
            r_k = np.sum(top_k_relevant) / num_relevant
            # Hit Rate@K
            hr_k = 1.0 if np.sum(top_k_relevant) > 0 else 0.0
            
            # NDCG@K
            dcg = 0.0
            for rank_idx, rel in enumerate(top_k_relevant):
                if rel:
                    dcg += 1.0 / np.log2(rank_idx + 2) # rank 0 is position 1 -> log2(2)=1
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, int(num_relevant))))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            
            precisions.append(p_k)
            recalls.append(r_k)
            ndcgs.append(ndcg)
            hit_rates.append(hr_k)
            
        prec_at_k = float(np.mean(precisions)) if precisions else 0.0
        rec_at_k = float(np.mean(recalls)) if recalls else 0.0
        ndcg_at_k = float(np.mean(ndcgs)) if ndcgs else 0.0
        hr_at_k = float(np.mean(hit_rates)) if hit_rates else 0.0
        
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec_at_k,
        'Recall@10': rec_at_k,
        'NDCG@10': ndcg_at_k,
        'HitRate@10': hr_at_k
    }

def train_model(model_name, model, data, epochs=35, lr=0.005, weight_decay=1e-5, device='cpu'):
    print(f"\n==================== Training {model_name} ====================")
    model = model.to(device)
    edge_index = data['edge_index'].to(device)
    movie_genres = data['movie_genre_feats'].to(device)
    train_pairs = data['train_pairs'].to(device)
    train_ratings = data['train_ratings'].to(device)
    val_pairs = data['val_pairs'].to(device)
    val_ratings = data['val_ratings'].to(device)
    test_pairs = data['test_pairs'].to(device)
    test_ratings = data['test_ratings'].to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_rmse': [],
        'val_mae': []
    }
    
    best_val_rmse = float('inf')
    best_weights = None
    start_time = time.time()
    
    u_tr = train_pairs[:, 0]
    m_tr = train_pairs[:, 1]
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. GNN Message Passing
        h_u, h_m = model(edge_index, movie_genres)
        
        # 2. Rating Prediction on all training pairs
        preds = model.predict_rating(h_u[u_tr], h_m[m_tr])
        loss = criterion(preds, train_ratings)
        
        # 3. Backward & Optimizer Step
        loss.backward()
        optimizer.step()
        
        avg_train_loss = loss.item()
        
        # 4. Validation
        val_metrics = evaluate_metrics(model, edge_index, movie_genres, val_pairs, val_ratings, data['n_users'], data['n_movies'])
        val_rmse = val_metrics['RMSE']
        val_mae = val_metrics['MAE']
        scheduler.step(val_rmse)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Best Val RMSE: {best_val_rmse:.4f}")
    
    # Load best weights for final test evaluation
    model.load_state_dict(best_weights)
    model.eval()
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_weights, f"checkpoints/{model_name.lower()}_best.pt")
    
    # Test evaluation
    test_metrics = evaluate_metrics(model, edge_index, movie_genres, test_pairs, test_ratings, data['n_users'], data['n_movies'])
    test_metrics['TrainingTime_sec'] = round(total_time, 2)
    test_metrics['BestValRMSE'] = round(best_val_rmse, 4)
    print(f"Test Metrics for {model_name}:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
        
    return test_metrics, history

def run_all_experiments():
    # Load data
    data_path = 'movielens_graph_data.pt'
    if not os.path.exists(data_path):
        from data_loader import prepare_movielens_data
        data = prepare_movielens_data()
        torch.save(data, data_path)
    else:
        data = torch.load(data_path, weights_only=False)
        
    n_users = data['n_users']
    n_movies = data['n_movies']
    genre_dim = data['movie_genre_feats'].shape[1]
    
    print(f"Dataset: {n_users} users, {n_movies} movies, {genre_dim} genre features.")
    
    # Instantiate models
    torch.manual_seed(42)
    np.random.seed(42)
    
    models = {
        'GCN': GCNRecommender(n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64),
        'GraphSAGE': GraphSAGERecommender(n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64),
        'GAT': GATRecommender(n_users, n_movies, genre_dim, embedding_dim=64, hidden_dim=64, heads=4)
    }
    
    results = {}
    histories = {}
    
    for name, model in models.items():
        # Reset seed before each model training for deterministic initialization and fair comparison
        torch.manual_seed(42)
        np.random.seed(42)
        metrics, history = train_model(name, model, data, epochs=35, lr=0.005)
        results[name] = metrics
        histories[name] = history
        
    # Save results to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    with open('training_histories.json', 'w') as f:
        json.dump(histories, f, indent=4)
        
    print("\n==================== COMPARISON SUMMARY ====================")
    df_results = pd.DataFrame(results).T
    print(df_results[['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10', 'HitRate@10', 'TrainingTime_sec']])
    return results, histories

if __name__ == '__main__':
    run_all_experiments()
