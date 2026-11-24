"""
Streamlit Web Interface for Movie Recommendation System
Powered by GCN, GraphSAGE, and GAT trained on Hugging Face MovieLens.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommender import MovieRecommenderSystem

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CineGraph: GNN Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .movie-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.75rem;
        background-color: #EEF2F6;
        color: #334155;
    }
    .winner-tag {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return MovieRecommenderSystem()

@st.cache_data
def load_results():
    with open('experiment_results.json', 'r') as f:
        results = json.load(f)
    with open('training_histories.json', 'r') as f:
        histories = json.load(f)
    return results, histories

rec = load_recommender()
results, histories = load_results()

# Sidebar
st.sidebar.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.markdown("## 🎬 **CineGraph GNN Recommender**")
st.sidebar.markdown(
    "A Graph Neural Network recommendation engine trained on the **real Hugging Face MovieLens dataset** "
    "(`ashraq/movielens_ratings`), benchmarking **GCN**, **GraphSAGE**, and **GAT**."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Settings")
selected_user_id = st.sidebar.number_input("Select User ID (0 to 1199):", min_value=0, max_value=rec.n_users - 1, value=42, step=1)
model_choice = st.sidebar.selectbox("Active Recommendation Model:", ["GraphSAGE (Best)", "GCN", "GAT"], index=0)
model_name = "GraphSAGE" if "GraphSAGE" in model_choice else ("GCN" if "GCN" in model_choice else "GAT")
top_k_count = st.sidebar.slider("Number of Recommendations:", min_value=3, max_value=20, value=8, step=1)

# Header
st.markdown('<div class="main-title">🎬 CineGraph: GNN Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empirical Benchmark of GCN, GraphSAGE, and GAT on Hugging Face Real-World Graph Data</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Personalized Recommendations", 
    "📊 Model Comparison & Metrics", 
    "🧠 In-Depth Analysis: Why GraphSAGE Won",
    "🔍 Movie Similarity Explorer"
])

# ----------------- TAB 1: RECOMMENDATIONS -----------------
with tab1:
    st.markdown(f"### 👤 User #{selected_user_id} Profile & Recommendation Feed")
    
    col_u1, col_u2 = st.columns([1, 2])
    
    with col_u1:
        st.markdown("#### 📜 User's Top Rated Movies")
        history = rec.get_user_history(selected_user_id)
        if history:
            st.caption(f"User #{selected_user_id} has rated {len(history)} movies in the graph.")
            for item in history[:5]:
                genres_str = " • ".join(item['genres'][:3])
                st.markdown(f"⭐ **{item['rating']:.1f}** | **{item['title']}**")
                st.caption(f"_{genres_str}_")
        else:
            st.info("No rating history available for this user.")
            
    with col_u2:
        st.markdown(f"#### 🎯 Top-{top_k_count} Recommendations ({model_choice})")
        recs = rec.recommend_for_user(selected_user_id, model_name=model_name, top_k=top_k_count)
        
        # Display recommendations in responsive grid columns
        grid_cols = st.columns(4)
        for i, item in enumerate(recs):
            col = grid_cols[i % 4]
            with col:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                poster_url = item.get('poster')
                if poster_url and isinstance(poster_url, str) and poster_url.startswith('http'):
                    st.image(poster_url, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    
                st.markdown(f"**{item['title']}**")
                st.markdown(f"✨ **Score: {item['predicted_rating']:.2f} / 5.0**")
                
                # Genre badges
                genres_html = " ".join([f'<span class="badge">{g}</span>' for g in item['genres'][:2]])
                st.markdown(genres_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
    st.markdown("---")
    st.markdown("### 🔄 Cross-Model Comparison for User #{}".format(selected_user_id))
    st.caption("Compare the top recommendations generated by all 3 GNN architectures for this specific user:")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    for c, m in zip([m_col1, m_col2, m_col3], ["GCN", "GraphSAGE", "GAT"]):
        with c:
            is_best = " ⭐ (Best Performer)" if m == "GraphSAGE" else ""
            st.markdown(f"#### {m}{is_best}")
            m_recs = rec.recommend_for_user(selected_user_id, model_name=m, top_k=5)
            for idx, r in enumerate(m_recs, 1):
                st.markdown(f"**{idx}. {r['title']}**")
                st.caption(f"Score: `{r['predicted_rating']:.2f}` | {', '.join(r['genres'][:2])}")

# ----------------- TAB 2: MODEL COMPARISON -----------------
with tab2:
    st.markdown("### 🏆 Comprehensive Model Evaluation on Test Split (20,030 interactions)")
    st.caption("All models were trained under identical conditions: same split, same seed, same learning rate (0.005), and 35 epochs.")
    
    # Metrics Table
    df_metrics = pd.DataFrame(results).T
    df_display = df_metrics[['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10', 'HitRate@10', 'TrainingTime_sec']].copy()
    
    # Highlight best in each column
    st.dataframe(
        df_display.style.highlight_min(subset=['RMSE', 'MAE', 'TrainingTime_sec'], color='#DCFCE7')
                  .highlight_max(subset=['Precision@10', 'Recall@10', 'NDCG@10', 'HitRate@10'], color='#DCFCE7')
                  .format("{:.4f}"),
        use_container_width=True
    )
    
    st.markdown("""
    > **Highlight Guide:** Green cells represent the winning model for each evaluation metric.
    """)
    
    # Visual Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 📉 Rating Prediction Error (Lower is Better)")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        x = np.arange(len(df_metrics))
        width = 0.35
        
        ax1.bar(x - width/2, df_metrics['RMSE'], width, label='RMSE', color='#3B82F6')
        ax1.bar(x + width/2, df_metrics['MAE'], width, label='MAE', color='#93C5FD')
        
        ax1.set_ylabel('Error')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_metrics.index)
        ax1.set_ylim(0, 1.2)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig1)
        
    with col_chart2:
        st.markdown("#### 📈 Top-10 Ranking Quality (Higher is Better)")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        width = 0.25
        
        ax2.bar(x - width, df_metrics['Precision@10'], width, label='Precision@10', color='#10B981')
        ax2.bar(x, df_metrics['Recall@10'], width, label='Recall@10', color='#06B6D4')
        ax2.bar(x + width, df_metrics['NDCG@10'], width, label='NDCG@10', color='#8B5CF6')
        
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_metrics.index)
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig2)
        
    # Training Convergence Plot
    st.markdown("#### ⏱️ Training Convergence Curves Across Epochs")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 3.8))
    
    colors = {'GCN': '#3B82F6', 'GraphSAGE': '#10B981', 'GAT': '#F59E0B'}
    for m in ['GCN', 'GraphSAGE', 'GAT']:
        epochs = histories[m]['epoch']
        ax3a.plot(epochs, histories[m]['train_loss'], label=m, color=colors[m], linewidth=2)
        ax3b.plot(epochs, histories[m]['val_rmse'], label=m, color=colors[m], linewidth=2)
        
    ax3a.set_title('Training MSE Loss')
    ax3a.set_xlabel('Epoch')
    ax3a.set_ylabel('Loss')
    ax3a.legend()
    ax3a.grid(True, linestyle='--', alpha=0.5)
    
    ax3b.set_title('Validation RMSE')
    ax3b.set_xlabel('Epoch')
    ax3b.set_ylabel('RMSE')
    ax3b.legend()
    ax3b.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig3)

# ----------------- TAB 3: ANALYSIS -----------------
with tab3:
    st.markdown("### 🧠 Technical Analysis: Which Model Performs Best and Why?")
    
    st.markdown("""
    #### 🥇 Verdict: **GraphSAGE is the Clear Winner**
    
    Across both **rating prediction accuracy** and **top-K ranking quality**, **GraphSAGE** outperforms GCN and GAT:
    * **Lowest RMSE (0.8856)**: 8.3% improvement over GCN (0.9655) and 10.1% improvement over GAT (0.9854).
    * **Lowest MAE (0.6570)**: 11.2% improvement over GCN (0.7402) and 13.9% improvement over GAT (0.7631).
    * **Highest NDCG@10 (0.8213)**: Demonstrates the best capability to rank top-rated items at higher positions.
    * **Highest Recall@10 (75.73%) & Precision@10 (66.49%)**.
    * **Fastest Training Time (31.05s)**: GraphSAGE converges faster than GCN (34.16s) and GAT (42.47s).
    
    ---
    
    #### 🔬 Why GraphSAGE Performs Best: Key Architectural Advantages
    
    ##### 1. Self-Feature Preservation vs. Over-Smoothing (GCN's Weakness)
    * **The GCN Mechanism**: GCN computes node representations using symmetric normalized Laplacian aggregation:
      $$h_i^{(l+1)} = \\sigma\\left( \\sum_{j \\in \\mathcal{N}(i) \\cup \\{i\\}} \\frac{1}{\\sqrt{\\tilde{d}_i \\tilde{d}_j}} W h_j^{(l)} \\right)$$
      In a bipartite user-movie graph, high-degree nodes (popular blockbuster movies and power users) have hundreds of incident edges. Because GCN mixes the central node's representation with its neighbors under a single shared linear transformation $W$, the central node's unique preference signal gets heavily diluted (*over-smoothed*) by the high-volume neighbor representations.
    * **The GraphSAGE Mechanism**: GraphSAGE separates the transformation of the central node from its neighborhood:
      $$h_i^{(l+1)} = \\sigma\\left( W_1 h_i^{(l)} + W_2 \\cdot \\text{AGG}\\left(\\{h_j^{(l)}, j \\in \\mathcal{N}(i)\\}\\right) \\right)$$
      By keeping $W_1$ dedicated to the central node's identity and $W_2$ dedicated to collaborative neighborhood context, GraphSAGE preserves the user's specific latent taste and prevents representation collapse.
      
    ##### 2. Inductive Generalization on Bipartite Relational Structure
    * Collaborative filtering graphs exhibit extreme variance in node degree. GraphSAGE's mean aggregation naturally normalizes neighborhood scale without relying on fixed symmetric degrees like GCN, giving much better inductive stability when predicting across validation and test sets.
    
    ##### 3. Why GAT Underperformed Relative to GraphSAGE Here
    * **Over-parameterization & Attention Overfitting**: GAT introduces parameter-heavy dynamic attention vectors for every head ($a^\\top [W h_i \\| W h_j]$). In an explicit rating recommendation setting with limited input feature diversity (node embeddings + genre vectors), the attention weights easily overfit to noisy individual ratings during training.
    * **Optimization & Training Dynamics**: In bipartite graphs, computing pairwise softmax normalizations across massive neighbor sets creates sharper loss landscapes and slower gradient propagation, leading GAT to settle in suboptimal local minima (Train Loss ~0.96 vs GraphSAGE ~0.78).
    * **Higher Latency**: GAT requires 42.47s to train compared to GraphSAGE's 31.05s due to per-edge attention coefficients and multi-head concatenation overhead.
    """)

# ----------------- TAB 4: SIMILARITY EXPLORER -----------------
with tab4:
    st.markdown("### 🔍 Latent Space Movie Similarity Explorer")
    st.caption("Inspect how the Graph Neural Network positions movies in embedding space based on both collaborative ratings and genre topology.")
    
    all_movie_titles = [rec.movie_meta[i]['title'] for i in range(rec.n_movies) if i in rec.movie_meta]
    selected_movie_title = st.selectbox("Select a Movie to Find Similar Titles:", sorted(all_movie_titles), index=0)
    
    # Find movie index
    target_idx = None
    for idx, meta in rec.movie_meta.items():
        if meta.get('title') == selected_movie_title:
            target_idx = idx
            break
            
    if target_idx is not None:
        target_meta = rec.movie_meta[target_idx]
        t_col1, t_col2 = st.columns([1, 4])
        with t_col1:
            if target_meta.get('poster') and target_meta['poster'].startswith('http'):
                st.image(target_meta['poster'], width=150)
        with t_col2:
            st.markdown(f"#### {target_meta['title']}")
            st.markdown(f"**Genres:** {', '.join(target_meta['genres'])}")
            st.caption(f"Internal Graph Movie Index: `{target_idx}` | MovieID: `{target_meta['movie_id']}`")
            
        st.markdown("---")
        st.markdown("#### 🌟 Most Similar Movies in Learned Embedding Space (Cosine Similarity):")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        for col, m_name in zip([sim_col1, sim_col2, sim_col3], ["GraphSAGE", "GCN", "GAT"]):
            with col:
                st.markdown(f"##### {m_name}")
                similars = rec.get_similar_movies(target_idx, model_name=m_name, top_k=5)
                for s in similars:
                    st.markdown(f"• **{s['title']}**")
                    st.caption(f"Similarity: `{s['similarity']:.3f}` | {', '.join(s['genres'][:2])}")
