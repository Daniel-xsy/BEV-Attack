# BEV Attack

## About

3D visual perception tasks, including 3D detection and map segmentation are essential for autonomous driving systems. One promising method is BEV perception, which perform detection or segmentaion from bird's eye view. Large numbers of works have demonstrated the vulnerability of 3D detection models either using LiDAR or camera images. However, the robustness of BEV model is less exploited. In this work, we try to study the robustness of BEV model under different adversarial attacks or common corruptions.

## Updates

- [2022.09.23] - Project starts

## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)
- [TODO List](#todo-list)

## Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n bev python=3.8 -y
conda activate bev
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install timm.**
```shell
pip install timm
```

## Data Preparation

Coming soon.

## Getting Started

Coming soon.

## TODO List
- [x] Intial release.
- [ ] Build attack baseline