# 🎬 CineGraph: GNN Movie Recommendation System
### Benchmarking GCN, GraphSAGE, and GAT on Hugging Face MovieLens Data

This project implements, trains, and evaluates three fundamental Graph Neural Network (GNN) architectures—**GCN (Graph Convolutional Network)**, **GraphSAGE (Sample and Aggregate)**, and **GAT (Graph Attention Network)**—for collaborative movie recommendation on real-world interaction data from Hugging Face (`ashraq/movielens_ratings`).

The system includes a modern, interactive web application built with Streamlit for exploring personalized movie recommendations, analyzing model performance, comparing recommendations across models, and inspecting learned embedding similarities.

---

## 📊 1. Empirical Results & Comparative Benchmark

All three models were trained and evaluated on the exact same dataset splits (80% Train, 10% Validation, 10% Test) using identical hyperparameter protocols (Seed 42, 35 Epochs, Learning Rate 0.005, Weight Decay $1 \times 10^{-5}$, embedding dimension $d=64$).

| Model | Test RMSE (↓) | Test MAE (↓) | Precision@10 (↑) | Recall@10 (↑) | NDCG@10 (↑) | HitRate@10 (↑) | Training Time (s) (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GraphSAGE** 🏆 | **0.8856** | **0.6570** | **0.6649** | **0.7573** | **0.8213** | 0.9975 | **31.05s** |
| **GCN** | 0.9655 | 0.7402 | 0.6411 | 0.7340 | 0.7860 | **0.9983** | 34.16s |
| **GAT** | 0.9854 | 0.7631 | 0.6607 | 0.7554 | 0.8019 | 0.9958 | 42.47s |

* **Rating Accuracy Winner**: **GraphSAGE** achieves an **8.3% lower RMSE** than GCN and **10.1% lower RMSE** than GAT.
* **Top-10 Ranking Winner**: **GraphSAGE** achieves the highest **NDCG@10 (0.8213)**, **Recall@10 (75.73%)**, and **Precision@10 (66.49%)**.
* **Computational Efficiency Winner**: **GraphSAGE** is the fastest to converge (**31.05s** total runtime).

---

## 🧠 2. Which Model Performs Best and Why?

### **Verdict: GraphSAGE is the Clear Winner**

Across both explicit rating regression ($[0.5, 5.0]$ scale) and implicit ranking quality (Top-10 recommendations), **GraphSAGE** significantly outperforms GCN and GAT.

### **In-Depth Theoretical and Empirical Reasons:**

#### 1. Self-Feature Preservation vs. Neighborhood Over-Smoothing (GCN's Dilemma)
* **GCN's Mathematical Formulation**:
  $$h_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} W h_j^{(l)} \right)$$
  In bipartite recommendation graphs, node degree variance is immense (blockbuster movies and power users have hundreds of connections, while niche movies have few). In GCN, self-connections are simply added to the adjacency matrix ($\tilde{A} = A + I_N$), and both self-features and neighbor features pass through the **same** transformation weight matrix $W$. Consequently, for high-degree nodes, the node's own representation is overwhelmingly diluted by the summation of its neighbors, leading to severe **over-smoothing** and representation collapse.
* **GraphSAGE's Solution**:
  $$h_i^{(l+1)} = \sigma\left( W_1 h_i^{(l)} + W_2 \cdot \text{AGG}\left(\{h_j^{(l)}, j \in \mathcal{N}(i)\}\right) \right)$$
  GraphSAGE explicitly separates the transformation of the central node ($W_1$) from the aggregated neighborhood context ($W_2$). This decoupling preserves the user's specific latent identity while simultaneously incorporating collaborative neighborhood taste signals.

#### 2. Robust Inductive Aggregation on Bipartite Topology
* A user's preference is influenced by the movies they watched, but they are not merely the literal average of those movies. GraphSAGE's mean aggregator normalizes the neighborhood scale independently of fixed degree distributions, providing greater stability when generalizing to test edges.

#### 3. Why GAT Underperformed Relative to GraphSAGE Here
* **Attention Overfitting on Dense Rating Topologies**: GAT calculates dynamic edge attention coefficients:
  $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [W h_i \,\|\, W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [W h_i \,\|\, W h_k]\right)\right)}$$
  In collaborative filtering graphs where node features are initialized from latent embeddings and sparse genre vectors, GAT's multi-head attention introduces many extra parameters. Without auxiliary multimodal node features (e.g. rich text or images), dynamic attention easily overfits to noisy rating variances, causing optimization to plateau at suboptimal local minima (Train Loss 0.96 vs GraphSAGE 0.78).
* **Latency & Memory Footprint**: GAT computes pairwise attention softmax normalizations across hundreds of thousands of edges per epoch, resulting in a 36.8% runtime overhead compared to GraphSAGE (42.47s vs 31.05s).

---

## 🛠️ 3. System Architecture & Dataset

### **Dataset: Real MovieLens via Hugging Face**
* Source: `ashraq/movielens_ratings`
* Total Interactions: 200,295 real user-movie ratings across 1,200 unique users and 1,500 unique movies.
* Edge Splits: 160,236 Train edges, 20,029 Validation edges, 20,030 Test edges.
* Metadata: Real movie titles, 19 genre categories (Action, Sci-Fi, Drama, etc.), and IMDb/TMDb poster URLs.

### **Graph Formulation**
* **Graph Structure**: Bipartite undirected graph $\mathcal{G} = (\mathcal{V}_U \cup \mathcal{V}_M, \mathcal{E})$.
* **Node Representations**:
  * Users: Learnable embedding matrix $E_U \in \mathbb{R}^{N_U \times d}$ ($d=64$).
  * Movies: Learnable embedding matrix $E_M \in \mathbb{R}^{N_M \times d}$ combined with linear projection of 19-dimensional multi-hot genre vectors $W_g G_M$.
* **Predictor Head**:
  $$\hat{y}_{u,m} = \sigma\left(\text{MLP}([h_u \,\|\, h_m \,\|\, h_u \odot h_m])\right) \times 4.5 + 0.5$$
  The Hadamard product $h_u \odot h_m$ injects a symmetric bilateral alignment signal, while the concatenation preserves asymmetric capacity.

---

## 🚀 4. How to Run

### **1. Launch the Streamlit Web Application**
The web app is already running and accessible locally at:
👉 **`http://localhost:8501`**

To launch manually:
```bash
conda run -n pytgpu streamlit run app.py --server.port 8501
```

### **2. Re-train & Benchmark Models**
To re-run data processing and training from scratch:
```bash
# 1. Process dataset from Hugging Face
conda run -n pytgpu python data_loader.py

# 2. Train and evaluate GCN, GraphSAGE, and GAT
conda run -n pytgpu python train_eval.py
```

---

## 🖥️ 5. Web Interface Highlights
* **Personalized Recommendations**: Select any User ID (0–1199), view their past favorite movies, select the GNN model, and generate Top-K movie recommendations with full poster art and predicted ratings.
* **Side-by-Side Model Comparison**: Compare recommendations generated by GCN, GraphSAGE, and GAT for the same user simultaneously.
* **Evaluation Dashboard**: Interactive tables, bar plots, and training/validation loss curves.
* **Movie Similarity Explorer**: Search any movie in the database and find its nearest neighbors via cosine similarity in the learned GNN embedding space.
