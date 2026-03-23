
# EllipticGuard: Graph Deep Learning for Bitcoin Illicit Activity Detection

## Team Members

Ran Li

Shaoyang Zhou

Rafael Miksian Magaldi

Prakash Singh

Tinghao Huang

## Project Description

Detecting illicit activity (such as fraud) in Bitcoin transactions is a major challenge, as effective detection can prevent significant financial losses and legal risks. In this repository, we detect illicit transactions in Bitcoin using a range of modeling approaches, from classical statistical models to graph neural networks (GNNs) and hybrid methods that combine both. Our goal is to compare how effectively these approaches identify illicit transactions in complex transaction networks.

## Dataset

We use the Elliptic Bitcoin dataset, a temporal directed graph of Bitcoin transactions with 203,769 nodes and 234,355 edges, where nodes represent transactions and edges represent the flow of funds between them.

Each node has 166 numeric features (referred to as tabular features or node-only features):

- **Feature 1 (time step)**: a discrete index from 1 to 49, ordered chronologically. Transactions within the same time step occur at nearly the same time on the Bitcoin network.
- **Local features (next 93)**: transaction-level attributes such as number of inputs and outputs, transaction fees, and output volume.
- **Aggregate features (remaining 72)**: neighborhood-level statistics computed from neighboring nodes' feature values, such as max, min, standard deviation, and correlation of input and output counts, fees, and volumes across one-hop neighbors.


Data is stored under `Dataset`, where `raw` contains the original CSV files and `processed` contains PyTorch tensors and other preprocessed artifacts generated from them. 


## EDA

The notebooks in the `eda` folder explain the data before modeling:

- `eda/EDA.ipynb`: Studies labels and drift over time, split coverage, topology (degrees, per–time-step graphs, mixing/homophily), labeled-neighborhood diagnostics, and basic feature QC (including time-step leakage checks).

- `eda/data_visualization.ipynb`: Builds a PyG graph from processed tensors, visualizes local subgraphs around illicit nodes (NetworkX). It also does some light EDA (labels over time, correlation heatmap).

- `eda/Basic_study_Prakash.ipynb`: Performs structural graph analysis, class summaries, and connectivity statistics.

- `eda/elliptic_eda_preprocessing.ipynb`: Loads the packaged processed transaction graph, checks split coverage, and visualizes label and local-structure patterns on the graph.

- `eda/EDA_KS_GraphSignal.ipynb`: Studies KS-based feature separation and graph-signal behavior under the time-forward split.

## Modeling Methods

### Time-forward split

Since there is a darknet market shutdown at time step 43, we drop time steps 43 and beyond to evaluate our models in a stable regime. For time steps 1–42, we use a strict time-forward split: 1–32 for training, 33–37 for validation, and 38–42 for testing. 

### Evaluation metrics

We use PR-AUC, ROC-AUC, reliability curves, precision, recall, Brier score, and training time/total compute cost across our notebooks, but PR-AUC is the primary evaluation metric. This is because the dataset is highly imbalanced, and PR-AUC better reflects performance on the minority (illicit) class.

### Baseline models
These notebooks cover baseline statistical models trained on tabular features.

- `baseline_models/Baseline_models_32-5-5_split_Prakash.ipynb`: Trains and compares class-weighted L2 logistic regression, linear SGD (logistic loss), random forest, gradient-boosted trees (XGBoost, LightGBM, etc.), and a MLPClassifier.

- `baseline_models/Baseline-Regularized Logistic Regression.ipynb`: Fits logistic regression, then tunes L2-regularized logistic regression with `RandomizedSearchCV`, and plots PR/ROC curves and coefficient-based feature importance.

- `baseline_models/Tree_based_models.ipynb`: Hyperparameter-tuned `DecisionTreeClassifier` followed by `RandomForestClassifier` on tabular features, with PR/ROC analysis and an optional tree plot.

- `baseline_models/Baseline_ET600.ipynb`: Trains an **Extra Trees** classifier (600 trees, “ET-600”) and exports metrics plus prediction outputs.

Here is a summary of model performance:

| Model Name | Test PR-AUC |
| --- | --- |
| Logistic Regression | `TBD` |
| Linear SGD | `TBD` |
| Random Forest | 0.8911 |
| XGBoost| `TBD` |
| LightGMB | `TBD`  |
| ET-600 | `0.8965` |

### GNN models

As baseline models use only tabular features and do not leverage graph information, we study several graph-based models for illicit transaction detection. These include **Graph Convolutional Network (GCN)**, which aggregates information from neighboring nodes through graph convolutions; **GraphSAGE** and its variants, which learn node representations by sampling and aggregating neighborhood features; **Graph Attention Network (GAT)**, which uses attention to weight neighbors differently; **General, Powerful, Scalable Graph Transformer(GraphGPS)**, which blends local message passing with transformer-style global attention; **Approximate Personalized Propagation of Neural Predictions (APPNP)**, which combines neural predictions with personalized propagation over the graph; and residual or propagation-based baselines such as **Scalable Inception Graph Neural Network (SIGN)**, **LinearResidual**, **MLP-Residual**, and **MLP-LayerNorm**, which test how far simpler architectures can go with appropriate feature propagation or skip connections.

These models are explored across the following notebooks:

- `gnn/GNN_models_32-5-5_split.ipynb`: GCN, GraphSAGE, GraphSAGE-Max, GraphSAGE-JK, GraphSAGE-Wide, GraphSAGE-JK-Wide, GAT, MLP-Residual, MLP-LayerNorm, and APPNP.
- `gnn/gnn_models.ipynb`: GCN, GraphSAGE, GraphGPS.
- `gnn/gat_model.ipynb` and `gnn/GAT model.ipynb`: Focused experiements on GAT.
- `gnn/Directed_Residual_GNNs.ipynb`: SIGN, APPNP, LinearResidual.
Among the currently uploaded standalone graph models, the strongest later graph results come from `gnn/Directed_Residual_GNNs.ipynb`. This notebook trains directed residual graph models that start from a strong tabular baseline, use the directed transaction graph to add local context, and predict only a small correction to the baseline score. The strongest results in this family are Directed SIGN residual and Directed APPNP residual.

Here is a summary of model performance:

| Model Name | Test PR-AUC |
| --- | --- |
| GCN | 0.6022 |
| GAT | `TBD` |
| GraphSage | 0.8153 |
| GraphGPS | 0.8252 |
| APPNP | 0.9150 |
| SIGN | 0.9154 |


### Hybrid models (Graph-Tree Integration):

We explore hybrid model structures that combine graph information with tabular features. In particular, we study graph–tree integration through two approaches:

**Approach 1**: Use embeddings from an upstream GNN, concatenated with tabular features to train downstream tree-based models.  

Recall that GNNs naturally learn node embeddings, representing each node as a vector in $\mathbb{R}^N$. These embeddings map nodes with similar structural roles or connectivity patterns to nearby points in the embedding space. When effective, they can serve as additional features alongside the original node attributes. By augmenting statistical models with these embeddings, we aim to improve predictive performance. However, three challenges arise: 

- **Dimensionality trap**: large, unconstrained embeddings introduce topological noise that tree ensembles may overfit;

- **Over-smoothing**: graph mixing can wash out important local information;

- **Lack of end-to-end training**: the embedding matrix is optimized for the GNN classification layer, not for the downstream model.

The following notebooks address these challenges.

- `gnn/Pre_Shutdown_Hybrid_Matryoshkas.ipynb`: Addresses the first two challenges. In particular, **Matryoshka bottlenecks** train embeddings at multiple dimensions simultaneously, encouraging the model to compress essential structural information into low-dimensional representations and filter out topological noise. **Skip-GCN** introduces a residual pathway that preserves raw node features, helping prevent over-smoothing and retain important local information. Empirically, low-dimensional embeddings (e.g., 4D–16D) capture useful graph structure while avoiding overfitting, whereas larger embeddings degrade performance due to noise, confirming the dimensionality trap. The best hybrid models achieve performance comparable to strong tabular baselines while providing a more principled integration of graph information.

- `graph_tree_hybrid.ipynb`: Addresses the third challenge via end-to-end training using a **Differentiable Neural Decision Forest (DNDF)**: the final prediction layer of GraphGPS is replaced by a differentiable decision-tree head (`max_depth = 5`), where routing is probabilistic rather than deterministic so the module can be trained with back propagation. The model is trained with embedding dimension D = 16, producing a node embedding matrix shaped for that tree-like objective. Those embeddings are concatenated with the original tabular features to train Random Forest models with `max_depth = 5`, which outperform the same random forest trained on tabular features only. In `hypothesis_testing.ipynb`, we use paired t-tests across 36 random-forest configurations (hybrid vs. tabular-only) to show that the hybrid gain is a consistent improvement, not a one-off.

**Approach 2**: Use graph-aggregated features directly to train tree-based models.

This is implemented in `graph_non_gnn_models/GraphAgg_ET.ipynb`, which builds a two-stage, non-GNN graph-learning pipeline. First, a time-respecting Extra Trees anchor model generates transaction risk scores. Then those scores are converted into directed neighbor-risk aggregate features, including in/out degree, mean and max neighbor risk, and counts of high-risk neighbors. These graph-derived features are combined with local transaction features in a second Extra Trees model. The final model reaches **0.9050 test PR-AUC**, showing that directional graph aggregates can deliver strong graph-based gains even without a graph neural network.


| Model Name | Test PR-AUC |
| --- | --- |
| Matryoshka Hybrid | 0.8943 |
| Deep Neural Decision Forest | 0.8647 |
| Random Forest on Hybrid Features | 0.8982 |
| GraphAgg ET | 0.9050 |


### The Final Model

The best final result came from a **combination model** implemented in `final_combination_models/Final_Best_Combination.ipynb`. This model keeps the strongest top part from the GraphAgg ET model, uses a short **0.4 SIGN + 0.6 stack** blend in a narrow middle band, and then uses **SIGN** for the remaining tail. This preserves the strongest graph-aware tree predictions while improving the rest of the final ordering with later graph models.

| Model Name | Test PR-AUC |
| --- | --- |
| Final Combination Model | `0.9187` |

