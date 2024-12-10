## Sensitivity Analysis

### Figure 1
**The effect of three hyperparameters on the NDCG@10 for the Mafengwo dataset.**
![image](https://github.com/user-attachments/assets/62506323-466f-4526-831c-ce2f07a015c7)


### Figure 2
**The effect of the polynomial graph filters on the NDCG@10 for the Mafengwo dataset.**
![image](https://github.com/user-attachments/assets/0ad59346-11b4-4a77-bc2e-a10983b25e5f)

<br>

We analyze the sensitivity of Group-GF to variations in $\alpha$, $\beta$, and $s$, as well as polynomial graph filters on the **Mafengwo** dataset.
- Effect of $\alpha$: NDCG@10 improves with increasing $\alpha$ up to 0.3, but decreases beyond that, indicating overemphasis on group interactions harms individual preferences.
- Effect of $\beta$: NDCG@10 improves with $\beta$ up to 0.5, with performance plateauing beyond that, suggesting sufficient capture of unified interactions.
- Effect of $s$: NDCG@10 improves as $s$ increases to 0.7, but plateaus and degrades slightly beyond $s=0.8$, indicating a need for balance.
- Effect of polynomial graph filters: The optimal performance in NDCG@10 is achieved by applying second-order $f_1(\bar{P}_u)$, third-order $f_2(\bar{P}_{g})$, and third-order $f_3(\bar{P}_{\text{uni}})$ filters, which capture nuanced interaction patterns for improved accuracy.

<br>

---

### Figure 3
**The effect of three hyperparameters on the NDCG@10 for the CAMRa2011 dataset.**
![image](https://github.com/user-attachments/assets/b4d0acd8-35af-4283-9791-f6f315080162)


### Figure 4
**The effect of the polynomial graph filters on the NDCG@10 for the CAMRa2011 dataset.**
![image](https://github.com/user-attachments/assets/af1c0fb2-2c11-414e-841f-5c326bdbcb52)

<br>

We also conducted experiments on the **CAMRa2011** dataset.
