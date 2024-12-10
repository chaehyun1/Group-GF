## Comparison with Canonical Recommendation Methods

### Table 1
**1. Performance comparison between the six representative standard recommendation methods and the four group recommendation methods on the Mafengwo dataset in terms of NDCG@5 and NDCG@10.**

| **Method**                | **NDCG@5** | **NDCG@10** |
|---------------------------|------------|-------------|
| **Canonical recommendation methods** |            |             |
| Pop                       | 0.2169     | 0.2537      |
| NeuMF                     | 0.5699     | 0.6012      |
| NGCF                      | 0.6424     | 0.6857      |
| LightGCN                  | 0.6571     | 0.7012      |
| SGL                       | 0.7580     | 0.7729      |
| Turbo-CF                  | 0.7886     | 0.7917      |
| **Group recommendation methods** |            |             |
| GroupIM                   | 0.6078     | 0.6330      |
| CubeRec                   | 0.7574     | 0.7708      |
| ConsRec                   | 0.7692     | 0.7794      |
| **Group-GF**              | **0.8384** | **0.8451**  |

---

**2. Performance comparison between the six representative standard recommendation methods and the four group recommendation methods on the CAMRa2011 dataset in terms of NDCG@5 and NDCG@10.**

| **Method**            | **NDCG@5** | **NDCG@10** |
|-----------------------|------------|-------------|
| **Canonical methods**  |            |             |
| Pop                   | 0.2825     | 0.3302      |
| NeuMF                 | 0.3198     | 0.3982      |
| NGCF                  | 0.3946     | 0.4520      |
| LightGCN              | 0.3978     | 0.4586      |
| SGL                   | 0.3982     | 0.4596      |
| Turbo-CF              | 0.4111     | 0.4755      |
| **Group recommendation methods** |   |             |
| GroupIM               | 0.4310     | 0.4913      |
| CubeRec               | 0.4346     | 0.4935      |
| ConsRec               | 0.4358     | 0.4945      |
| **Group-GF**          | **0.4466** | **0.5030**  |

<br>

In this experiment, we adopt six canonical recommendation methods in a group recommendation context by using only group–item interactions.
- Group-GF exceeds all state-of-the-art canonical recommendation methods in accuracy, highlighting its strength in integrating group-level information effectively.
- Naïve methods like Pop show poor results, emphasizing the importance of leveraging both member-level and group-level interactions in recommendations.
- Some group methods perform worse than canonical methods that only use group-item interactions (e.g., GroupIM). This suggests that simply using member and group interactions isn't enough for better performance.
- The improvement of Group-GF is due to its use of augmented, unified item similarity graphs and multi-view graph filters, consolidating all member and group information effectively.
