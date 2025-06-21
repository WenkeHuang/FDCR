# Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated LearningAdd commentMore actions

> Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning,            
> Wenke Huang, Mang Ye, Zekun Shi, Guancheng Wan, Bo Du
> *NeurIPS, 2024*
> [Link]()

## Abstract

Backdoor attacks pose a serious threat to federated systems, where malicious clients optimize on the triggered distribution to mislead the global model towards a predefined target. Existing backdoor defense methods typically require either homogeneous assumption, validation datasets, or client optimization conflicts. In our work, we observe that benign heterogeneous distributions and malicious triggered distributions exhibit distinct parameter importance degrees. We introduce the Fisher Discrepancy Cluster and Rescale (FDCR) method, which utilizes Fisher Information to calculate the degree of parameter importance for local distributions. This allows us to reweight client parameter updates and identify those with large discrepancies as backdoor attackers. Furthermore, we prioritize rescaling important parameters to expedite adaptation to the target distribution, encouraging significant elements to contribute more while diminishing the influence of trivial ones. This approach enables FDCR to handle backdoor attacks in heterogeneous federated learning environments. Empirical results on various heterogeneous federated scenarios under backdoor attacks demonstrate the effectiveness of our method.


## Citation
```
@inproceedings{FDCR_NeurIPS24,
    title    = {Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning},
    author    = {Huang, Wenke and Ye, Mang and Shi, Zekun and Wan, Guancheng and Du, Bo and Tao, Dacheng},
    booktitle = {NeurIPS},
    year      = {2024}Add commentMore actions
}
```

## Relevant Projects
[3] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)][[Code](https://github.com/WenkeHuang/RethinkFL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]

[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]