# 🎬 CineGraph: Graph Neural Networks for Movie Recommendation
### Benchmarking GCN, GraphSAGE, and GAT on Hugging Face Real-World Graph Data

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/PyG-torch__geometric-3C2179?style=for-the-badge&logo=pyg&logoColor=white" alt="PyG" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Executive Summary

**CineGraph** is an end-to-end Graph Neural Network (GNN) collaborative filtering recommendation system. It implements, trains, and empirically benchmarks three landmark graph architectures—**GCN**, **GraphSAGE**, and **GAT**—on real-world interaction data from Hugging Face ([`ashraq/movielens_ratings`](https://huggingface.co/datasets/ashraq/movielens_ratings)).

The repository features:
1. **Mathematical implementations** of **GCN**, **GraphSAGE**, and **GAT** tailored for bipartite rating prediction and ranking.
2. **Controlled empirical benchmarking** across 200,295 interaction edges under identical seed, split, and training conditions.
3. **Comprehensive evaluation metrics**: Rating Accuracy (**RMSE**, **MAE**) and Ranking Quality (**Precision@10**, **Recall@10**, **NDCG@10**, **HitRate@10**).
4. **An interactive, responsive Streamlit Web App** featuring real movie posters, multi-model side-by-side comparisons, and embedding similarity search.

---

## 🏆 Benchmark Results

All models were evaluated on the held-out test split (**20,030 interactions**) under identical conditions (Random Seed `42`, 35 Epochs, Learning Rate `0.005`, Weight Decay $10^{-5}$, embedding dimension $d=64$).

| Model | Test RMSE (↓) | Test MAE (↓) | Precision@10 (↑) | Recall@10 (↑) | NDCG@10 (↑) | HitRate@10 (↑) | Training Time (s) (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **GraphSAGE** | **0.8856** | **0.6570** | **0.6649** | **0.7573** | **0.8213** | 0.9975 | **31.05s** |
| 🥈 **GCN** | 0.9655 | 0.7402 | 0.6411 | 0.7340 | 0.7860 | **0.9983** | 34.16s |
| 🥉 **GAT** | 0.9854 | 0.7631 | 0.6607 | 0.7554 | 0.8019 | 0.9958 | 42.47s |

* **Rating Accuracy**: GraphSAGE achieves an **8.3% lower RMSE** than GCN and **10.1% lower RMSE** than GAT.
* **Top-10 Ranking Performance**: GraphSAGE leads in **NDCG@10 (0.8213)**, **Recall@10 (75.73%)**, and **Precision@10 (66.49%)**.
* **Computational Efficiency**: GraphSAGE converges the fastest (**31.05s** runtime).

---

## 🧠 Architectural Comparison: Why Did GraphSAGE Win?

```
                 [ User Node (u) ]           [ Movie Node (m) ]
                         │                           │
                   Embeddings (64d)         Embeddings + Genres (64d)
                         │                           │
                         ▼                           ▼
                 ┌───────────────────────────────────────────┐
                 │       GNN Message Passing Layers         │
                 │   (GCN vs. GraphSAGE vs. GAT Conv)        │
                 └───────────────────────────────────────────┘
                                       │
                      Refined Latent Representations
                                  h_u, h_m
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │          Predictor Head (MLP)             │
                 │      [ h_u || h_m || (h_u ⊙ h_m) ]        │
                 └───────────────────────────────────────────┘
                                       │
                                       ▼
                          Predicted Rating [0.5 - 5.0]
```

### 1. GCN (Graph Convolutional Network) — *Kipf & Welling (2017)*
* **Mathematical Formulation**:
  $$h_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} W h_j^{(l)} \right)$$
* **Mechanics**: Computes a symmetric normalized Laplacian aggregation where neighbor features and self-features pass through the **same** weight transformation matrix $W$.
* **Why it fell behind**: In a bipartite recommendation graph, high-degree nodes (popular blockbuster movies and power users) have hundreds of incident edges. Because GCN averages the central node's embedding together with all its neighbors, the central node's unique preference signal is diluted (**over-smoothing**), leading to degraded personal ranking.

### 2. GraphSAGE (Sample and Aggregate) — *Hamilton et al. (2017)* 🏆
* **Mathematical Formulation**:
  $$h_i^{(l+1)} = \sigma\left( W_1 h_i^{(l)} + W_2 \cdot \text{AGG}\left(\{h_j^{(l)}, \forall j \in \mathcal{N}(i)\}\right) \right)$$
* **Mechanics**: Separates the central node transformation ($W_1$) from the collaborative neighborhood aggregation ($W_2$).
* **Why it won**:
  1. **Self-Feature Preservation**: A user's preference is informed by their history, but they are not merely the average of watched movies. By isolating $W_1$, GraphSAGE preserves the user's specific latent taste while absorbing collaborative neighborhood signals through $W_2$.
  2. **Inductive Stability**: The mean aggregator normalizes neighbor contributions independent of fixed degree distributions, providing superior generalization on unseen test pairs.

### 3. GAT (Graph Attention Network) — *Veličković et al. (2018)*
* **Mathematical Formulation**:
  $$h_i^{(l+1)} = \bigparallel_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(l)} \right)$$
  $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [W h_i \,\|\, W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [W h_i \,\|\, W h_k]\right)\right)}$$
* **Mechanics**: Dynamically computes edge attention coefficients $\alpha_{ij}$ using multi-head self-attention.
* **Why it underperformed relative to GraphSAGE here**:
  1. **Attention Overfitting**: In an explicit rating graph with sparse genre features, the parameterized attention vectors $\mathbf{a}^\top$ easily overfit to noisy rating variations, causing gradient optimization to plateau in suboptimal local minima.
  2. **Latency & Compute Overhead**: Computing pairwise attention softmax normalizations across hundreds of thousands of edges per epoch incurred a **36.8% runtime penalty** (42.47s vs 31.05s).

---

## 🖥️ Interactive Web Interface (Streamlit)

The system includes a modern, responsive web application built with Streamlit:

* **🎯 Personalized Recommendations**: Select any User ID (0–1199) to view their top-rated movies, select an active GNN model, and generate Top-K recommendations complete with **movie poster images**, titles, genres, and predicted ratings.
* **🔄 Side-by-Side Comparison**: Compare recommendations generated by GCN, GraphSAGE, and GAT for the exact same user side-by-side to observe agreement and diversity.
* **📊 Metrics Dashboard**: View the full evaluation table, interactive error bar charts (RMSE & MAE), ranking quality bars (NDCG@10, Precision@10, Recall@10), and epoch-by-epoch loss convergence curves.
* **🔍 Movie Similarity Explorer**: Search any movie in the database and find its closest neighbors via cosine similarity in the learned GNN latent space.

---

## 📁 Repository Structure

```plaintext
├── app.py                      # Streamlit interactive web interface
├── models.py                   # PyTorch Geometric GCN, GraphSAGE, and GAT models
├── data_loader.py              # Hugging Face MovieLens data ingestion & graph construction
├── train_eval.py               # Controlled training & multi-metric evaluation pipeline
├── recommender.py              # Inference engine & embedding similarity search
├── experiment_results.json     # Saved test evaluation metrics for all 3 models
├── training_histories.json     # Epoch-by-epoch loss & validation RMSE logs
├── requirements.txt            # Dependency specification
├── .gitignore                  # Git ignore rules
└── checkpoints/                # Best model checkpoints
    ├── graphsage_best.pt       # Trained GraphSAGE weights
    ├── gcn_best.pt             # Trained GCN weights
    └── gat_best.pt             # Trained GAT weights
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MoMassEg/recommendation-system-graph-neural-network.git
cd recommendation-system-graph-neural-network
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`**.

### 4. (Optional) Re-run Data Ingestion & Training
To prepare the dataset and re-train all models from scratch:
```bash
# Prepare dataset from Hugging Face
python data_loader.py

# Train & evaluate GCN, GraphSAGE, and GAT
python train_eval.py
```

---

## 📈 Evaluation Metrics Reference

* **RMSE (Root Mean Squared Error)**: Measures the standard deviation of rating prediction residuals. Heavily penalizes large rating errors.
* **MAE (Mean Absolute Error)**: Measures the average magnitude of absolute errors in rating prediction.
* **Precision@10**: Fraction of recommended movies in the top-10 that were relevant (rating $\ge 3.5$) in the ground truth test set.
* **Recall@10**: Fraction of all relevant test movies that appeared in the top-10 recommendations.
* **NDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality by discounting relevant items that appear at lower positions.
* **Hit Rate@10**: Proportion of users for whom at least one relevant movie was captured in the top-10.

---

## 📜 Dataset Citation

This project utilizes the MovieLens dataset hosted on Hugging Face:
```bibtex
@misc{movielens_ratings_hf,
  author = {Ashraq},
  title = {MovieLens Ratings with TMDb Metadata},
  year = {2022},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/ashraq/movielens_ratings}}
}
```

---

## 👨‍💻 Author & License

* **Author**: [Mohamed Shabaan (MoMassEg)](https://github.com/MoMassEg)
* **License**: MIT License
