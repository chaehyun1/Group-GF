## Sensitivity Analysis



We analyze the sensitivity of Group-GF to variations in $\alpha$, $\beta$, and $s$, as well as polynomial graph filters on the **Mafengwo** dataset.
- Effect of $\alpha$: NDCG@10 improves with increasing $\alpha$ up to 0.3, but decreases beyond that, indicating overemphasis on group interactions harms individual preferences.
- Effect of $\beta$: NDCG@10 improves with $\beta$ up to 0.5, with performance plateauing beyond that, suggesting sufficient capture of unified interactions.
- Effect of $s$: NDCG@10 improves as $s$ increases to 0.7, but plateaus and degrades slightly beyond $s=0.8$, indicating a need for balance.
- Effect of polynomial graph filters: The best performance in NDCG@10 is achieved by using second-order $f_1(\bar{P}_u)$, third-order $f_2(\bar{P}g)$, and third-order $f_3(\bar{P}{\text{uni}})$ filters, capturing nuanced interaction patterns for optimal accuracy.


We also conducted experiments on the CAMRa2011 dataset.
