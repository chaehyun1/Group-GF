## Analysis on Different Group Sizes

### Table 1
**1. Performance comparison of group recommendation methods for varying group sizes in terms of NDCG@5.**

| **Group size (top \%)** | **0-20\%** | **20-40\%** | **40-60\%** | **60-80\%** | **80-100\%** |
|-------------------------|-----------|------------|------------|------------|-------------|
| **GroupIM**             | 0.6597    | 0.7005     | 0.6047     | 0.5246     | 0.5016      |
| **CubeRec**             | 0.7584    | 0.7973     | 0.7714     | 0.6959     | 0.7652      |
| **ConsRec**             | 0.7620    | 0.7934     | 0.7806     | 0.7137     | 0.7715      |
| **Group-GF**            | **0.8504**| **0.8739** | **0.8415** | **0.8111** | **0.8101**  |


**2. Performance comparison of group recommendation methods for varying group sizes in terms of NDCG@10.**

| **Group size (top \%)** | **0-20\%** | **20-40\%** | **40-60\%** | **60-80\%** | **80-100\%** |
|------------------------|------------|-------------|-------------|-------------|--------------|
| **GroupIM**            | 0.7032     | 0.7343      | 0.6471      | 0.5615      | 0.5101       |
| **CubeRec**            | 0.7728     | 0.8070      | 0.7830      | 0.7177      | 0.7819       |
| **ConsRec**            | 0.7740     | 0.8028      | 0.7829      | 0.7295      | 0.7764       |
| **Group-GF**           | **0.8571** | **0.8783**  | **0.8441**  | **0.8195**  | **0.8199**   |

<br>

- We investigate the impact of group sizes on recommendation accuracy using the Mafengwo dataset. 
- The performance of Group-GF and other well-performing methods (GroupIM, CubeRec, and ConsRec) is summarized in Table 1, focusing on NDCG@5 and NDCG@10 across five group sizes (top 0-20%, 20-40%, 40-60%, 60-80%, and 80-100%).  

<br>

Findings:
1. For all methods, accuracy improves as group sizes decrease up to the top 40%, then steadily degrades. Smaller group sizes give individual members more influence, which makes it harder to utilize group-level interactions.
2. Despite such a challenge in recommendations for groups of small sizes, Group-GF consistently maintains high accuracy (> 0.8) across all group sizes, outperforming other methods. This suggests Group-GF is more robust to variations in group sizes.
