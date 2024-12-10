**Table 1: The statistics of large-scale synthetic datasets**
| # Users | # Items | # Groups | # U-I inter. | # G-I inter. |
|---------|---------|----------|--------------|--------------|
| 5k      | 10k     | 2.5k     | 1M           | 0.5M         |
| 10k     | 15k     | 5k       | 3M           | 1.5M         |
| 15k     | 20k     | 10k      | 6M           | 4M           |
| 20k     | 25k     | 15k      | 10M          | 7.5M         |
| 25k     | 30k     | 20k      | 15M          | 12M          |
| 27k     | 35k     | 22k      | 19.8M        | 15.4M        |


**Figure 1: Runtime performance of Group-GF across various sizes of synthetic datasets using CPU and GPU. Here, ‘GPU–OOM’ denotes an OOM issue on the GPU**
![image](https://github.com/user-attachments/assets/a0142aca-c377-4bb9-b265-021c4127dde5)  


To further validate the scalability of Group-GF, we run experiments by generating large-scale synthetic datasets having varying sizes (in terms of the number of groups, members, items, group-level interactions and member-level interactions). 
The statistics of each dataset are provided in Table 1. 
Figure 1 illustrates the runtime performance of Group-GF across various dataset sizes ranging from 0.5M to 15.4M group-level interactions. 
The experiments were conducted in various hardware environments, utilizing both CPU and GPU configurations. 
In this large-scale setup, while the use of GPU can deal with datasets with up to approximately 7.5M group-level interactions, the use of CPU shows impressive scalability, processing even the largest dataset with 15.4M group-level interactions in just 9m3s.
