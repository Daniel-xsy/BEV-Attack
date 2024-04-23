# BEV Attack

This is the codebase of our paper [On the Adversarial Robustness of Camera-based 3D Object Detection](https://arxiv.org/abs/2301.10766).

## About

In recent years, camera-based 3D object detection has gained widespread attention for its ability to achieve high performance with low computational cost. However, the robustness of these methods to adversarial attacks has not been thoroughly examined. In this study, we conduct the first comprehensive investigation of the robustness of leading camera-based 3D object detection methods under various adversarial conditions. Our experiments reveal five interesting findings: (a) the use of accurate depth estimation effectively improves robustness; (b) depth-estimation-free approaches do not show superior robustness; (c) bird's-eye-view-based representations exhibit greater robustness against localization attacks; (d) incorporating multi-frame benign inputs can effectively mitigate adversarial attacks; and (e) addressing long-tail problems can enhance robustness. We hope our work can provide guidance for the design of future camera-based object detection modules with improved adversarial robustness.

## Updates

- [2022.09.23] - Project starts
- [2022.09.29] - Build baseline attack

## Outline
- [BEV Attack](#bev-attack)
  - [About](#about)
  - [Updates](#updates)
  - [Outline](#outline)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Getting Started](#getting-started)
  - [TODO List](#todo-list)

## Installation

Please refer to [installation.md](docs/installation.md) for more details.

## Data Preparation

Please refer to [prepare_dataset.md](docs/prepare_dataset.md) for more details.

## Getting Started

Coming soon. Codebase will be organized when the author has time 💦.

### Brief information
For `BEVFormer`, `PETR`, and `DETR3D`, please refer to the [config](./mmdet_adv/projects/configs/attack) folder, and the [script](mmdet_adv/tools/attack.sh), just un-comment the attack at the end of each `config` you need to run.

For `BEVDet` and `BEVDepth`, please refer to the [config](./zoo/BEVDet/configs/attack) folder and the [script](./zoo/BEVDet/tools/attack.sh). 
> Offical `BEVDet` codebase has been changed and there might be some mismatch between this codebase and the current [official](https://github.com/HuangJunJie2017/BEVDet) one.

## TODO List
- [x] Intial release.
- [x] Build attack baseline.
- [ ] Add patch attack, reorganize code structure for more flexible usage.

## Citation
If you find this work helpful, please kindly consider citing the following:
```bib
@article{xie2023adversarial,
  title={On the Adversarial Robustness of Camera-based 3D Object Detection},
  author={Xie, Shaoyuan and Li, Zichao and Wang, Zeyu and Xie, Cihang},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}
```
