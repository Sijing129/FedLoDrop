# FedLoDrop: Federated LoRA with Dropout for Generalized LLM Fine-tuning

This repository contains the official implementation of our paper "FedLoDrop: Federated LoRA with Dropout for Generalized LLM Fine-tuning".

## Introduction
Fine-tuning (FT) large language models (LLMs) is crucial for adapting general-purpose models to specific tasks, enhancing accuracy and relevance with minimal resources. To further enhance generalization ability while reducing training costs, this paper proposes Federated LoRA with Dropout (FedLoDrop), a new framework that applies dropout to the rows and columns of the trainable matrix in Federated LoRA. A generalization error bound and convergence analysis under sparsity regularization are obtained, which elucidate the fundamental trade-off between underfitting and overfitting. The error bound reveals that a higher dropout rate increases model sparsity, thereby lowering the upper bound of pointwise hypothesis stability (PHS). While this reduces the gap between empirical and generalization errors, it also incurs a higher empirical error, which, together with the gap, determines the overall generalization error. On the other hand, though dropout reduces communication costs, deploying FedLoDrop at the network edge still faces challenges due to limited network resources. To address this issue, an optimization problem is formulated to minimize the upper bound of the generalization error, by jointly optimizing the dropout rate and resource allocation subject to the latency and per-device energy consumption constraints. To solve this problem, a branch-and-bound (B\&B)-based method is proposed to obtain its globally optimal solution. Moreover, to reduce the high computational complexity of the B\&B-based method, a penalized successive convex approximation (P-SCA)-based algorithm is proposed to efficiently obtain its high-quality suboptimal solution. Finally, numerical results demonstrate the effectiveness of the proposed approach in mitigating overfitting and improving the generalization capability.

## Framework Overview

Our proposed framework is illustrated as in Fig. 1. in the original paper.


## Environment
We recommend using a Conda environment to run the Python scripts for this project.


## How to use it
Following are the brief introductions of each file.
* fed_train_glue.py is the main file, including the dropout, LoRA, and federated training process.
* The task includes: `cola`, `mrpc`, `rte`, `stsb`, `sst2`, `qnli`.
* Model: `roberta-large` 

Notice that for the optimization algorithm about the network resources, this version just provides an example value.

## Citation

If you find our work useful for your research and projects, please consider citing our paper and starring our project!

```bibtex
@ARTICLE{11370828,
  author={Xie, Sijing and Wen, Dingzhu and You, Changsheng and Chen, Qimei and Bennis, Mehdi and Huang, Kaibin},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={FedLoDrop: Federated LoRA with Dropout for Generalized LLM Fine-tuning}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={LoRa;Overfitting;Adaptation models;Training;Costs;Matrix decomposition;Computational modeling;Performance evaluation;Convergence;Upper bound;Large Language Models;Federated Learning;Low-rank Adaptation;Dropout;Generalization Error},
  doi={10.1109/JSAC.2026.3660935}}

## Acknowledgement
We also thank the other authors for sharing their code, i.e., Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models. 
