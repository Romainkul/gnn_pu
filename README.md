# Master Thesis: "GNN on PU nodes"
### NNIF-GNN:
To do:
---
#### Matthias:
-  Implement GNN and MLP with nnPU, Imbalanced nnPU, and TED^n (base_test.py)
-  Implement a naive rf, XGBoost, and LR (base_test.py)
-  Implement the different methods for two-step (Spy, NNIF, and IF) (base_test.py)
-  Implement statistical tests (checck if possible) (stat_tests.py)
---
#### Romain:
-  Implement plot function for varying ratio of positives
-  Implement plot function for varying K (Look at 3D plot to show potential relation with pollution ratio)
-  Implement plot function for varying class prior (nnPU, Imb nnPU, Ours)
-  Implement table with best results:  NNIF (removal and relabeling), IF (removal and relabeling), Spy (SCAR)
-  Implement table with best results: Cluster-Sampling (ClusterGCN), NN-Sampling (SHINE), and Random Sampling (GraphSAGE)
-  Implement table with best results: GATConv, GCNConv, GINConv, and SAGEConv
