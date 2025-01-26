# Leveraging Member–Group Relations via Multi-View Graph Filtering for Effective Group Recommendation

This repository contains the official source code for Group-GF, presented at WWW 2025.

## Overview
![Image](https://github.com/user-attachments/assets/19fd605a-2128-462a-b16b-636425140fdb)  
Group recommender systems aim to provide precise recommendations tailored to a collection of members rather than individuals. To achieve this, capturing the intricate relationships between member-level and group-level interactions is essential for accurate group recommendations. To jointly deal with both member-level and group-level interactions, existing approaches often resort to hypergraph-based and self-supervised learning (SSL)-based methods. However, the aforementioned methods often involve complex hypergraph modeling or expensive model training costs for SSL, which can hinder their responsiveness to rapidly changing member preferences. To address these practical challenges, we propose Group-GF, the first attempt at group recommendations built upon graph filtering (GF). Our Group-GF method is composed of 1) the construction of three item similarity graphs exhibiting different viewpoints and 2) the optimal design of distinct polynomial graph filters that are hardware-friendly without costly matrix decomposition. Through extensive evaluations on benchmark datasets, Group-GF achieves not only state-of-the-art accuracy but also extraordinary runtime efficiency up to 1.55 seconds. In addition, we theoretically connect Group-GF’s filtering process to optimization with smoothness regularization, offering clearer interpretability of the model’s behavior.

## Installation  
Clone the repository and install the necessary dependencies:
```
pip install -r requirements.txt
```

```
python main.py --dataset="CAMRa2011" --verbose=0 --alpha=0.3 --beta=0.3 --power=0.9 --user_filter=1 --group_filter=2 --uni_filter=3
```
