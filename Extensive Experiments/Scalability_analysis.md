### Table 1
**The statistics of large-scale synthetic datasets**
| # Users | # Items | # Groups | # U-I inter. | # G-I inter. |
|---------|---------|----------|--------------|--------------|
| 5k      | 10k     | 2.5k     | 1M           | 0.5M         |
| 10k     | 15k     | 5k       | 3M           | 1.5M         |
| 15k     | 20k     | 10k      | 6M           | 4M           |
| 20k     | 25k     | 15k      | 10M          | 7.5M         |
| 25k     | 30k     | 20k      | 15M          | 12M          |
| 27k     | 35k     | 22k      | 19.8M        | 15.4M        |  

<br>

### Figure 1
**Runtime performance of Group-GF across various sizes of synthetic datasets using CPU and GPU. Here, ‘GPU–OOM’ denotes an OOM issue on the GPU**
![image](https://github.com/user-attachments/assets/a0142aca-c377-4bb9-b265-021c4127dde5)    

<br>

- Experiments validate Group-GF scalability using large-scale synthetic datasets with varying group, member, and item sizes, as well as interaction levels.
- Dataset statistics are detailed in Table 1.
- Runtime performance across dataset sizes (0.5M–15.4M group-level interactions) is shown in Figure 1.
- Experiments utilized both CPU and GPU configurations across different hardware setups.
- GPU handles datasets up to 7.5M group-level interactions, while CPU processes the largest dataset (15.4M interactions) in 9m3s.
