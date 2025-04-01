# [CVPR 2025] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation

This repository provides instructions for downloading LiSu, a synthetic LiDAR surface normal dataset generated using CARLA. It also includes a novel method that leverages spatiotemporal information from autonomous driving data to improve LiDAR surface normal estimation, demonstrated on both LiSu and the real-world Waymo dataset.

<div align="left">
    <a href="https://arxiv.org/abs/2503.08601" target="_blank">
        <img src="https://img.shields.io/badge/cs-2503.08601-b31b1b?logo=arxiv&logoColor=red" alt="Paper arXiv">
    </a>
    <a href="https://huggingface.co/datasets/dmalic/LiSu" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face">
    </a>
</div>

<details>
  <summary>Abstract</summary>
While surface normals are widely used to analyse 3D scene geometry, surface normal estimation from LiDAR point clouds remains severely underexplored. This is caused by the lack of large-scale annotated datasets on the one hand, and lack of methods that can robustly handle the sparse and often noisy LiDAR data in a reasonable time on the other hand. We address these limitations using a traffic simulation engine and present LiSu, the first large-scale, synthetic LiDAR point cloud dataset with ground truth surface normal annotations, eliminating the need for tedious manual labeling. Additionally, we propose a novel method that exploits the spatiotemporal characteristics of autonomous driving data to enhance surface normal estimation accuracy. By incorporating two regularization terms, we enforce spatial consistency among neighboring points and temporal smoothness across consecutive LiDAR frames. These regularizers are particularly effective in self-training settings, where they mitigate the impact of noisy pseudo-labels, enabling robust real-world deployment. We demonstrate the effectiveness of our method on LiSu, achieving state-of-the-art performance in LiDAR surface normal estimation. Moreover, we showcase its full potential in addressing the challenging task of synthetic-to-real domain adaptation, leading to improved neural surface reconstruction on real-world data.
</details>

<p align="center">
  <img src="docs/imgs/ex1.png" alt="LiDAR Surface Normal 1" width="49%">
  <img src="docs/imgs/ex2.png" alt="LiDAR Surface Normal 2" width="49%">
</p>

## Table of Contents
1. [Python Virtual Environment Setup](#venv_setup)
2. [Obtaining the LiSu Dataset](#lisu_dataset)
3. [Training](#training)
4. [TODOs](#todos)
5. [Citation](#citation)

## Python Virtual Environment Setup <a name="venv_setup"/>

### Recommended Specifications
- Python 3.10
- PyTorch 2.2.2
- CUDA 11.8

### Installation Steps
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install requirements
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
pip install flash-attn --no-build-isolation

pip install -r requirements.txt
```

## Obtaining the LiSu Dataset <a name="lisu_dataset"/>
LiSu dataset is freely accessibale on [🤗 Hugging Face](https://huggingface.co/datasets/dmalic/LiSu):
```bash
mkdir -p data && cd data
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/dmalic/LiSu
```
*Important Notes*:
- **Storage Requirements**: The dataset requires approximately 120GB of free disk space.
- **Alternative Access**: You can also access the dataset on Research Repository.

### Generate LiSu Dataset Extension
<div style="display: flex;">
  <a href="https://github.com/carla-simulator/carla/pull/8773" target="_blank">
    <img alt="GitHub pull request status" src="https://img.shields.io/github/status/s/pulls/carla-simulator/carla/8773?style=flat&label=CARLA%20PR">
  </a>
</div>

We support the expansion of the LiSu dataset with custom LiDAR configurations and driving routes. While our [CARLA pull request](https://github.com/carla-simulator/carla/pull/8773) is under review, you can leverage our Surface Normal LiDAR Sensor implementation from our [CARLA fork](https://github.com/malicd/carla/tree/malicd/lidar_surface_normals).

## Training <a name="training"/>
The training procedure consists of two main stages:
- Supervised training on the synthetic LiSu dataset.
- Self-supervised domain adaptation to real-world data (e.g., Waymo, easily extendable to other datasets).

### Supervised Training on LiSu
We trained our model on 8× NVIDIA A100 64GB GPUs. Below are instructions for both single-node and multi-node training.

#### Single-Node Multi-GPU Training
```bash
python main.py fit -c configs/lisu_ptv3_l1+gtv+tgtv+eikonal_fp16_fulltrainval.yaml --trainer.devices 8
```

#### Multi-Node Multi-GPU Training (SLURM Example)
For distributed training across multiple nodes, we provide a SLURM script example. This configures 2 nodes, each with 4 GPUs (adjustable for your cluster setup).
```bash
#!/bin/bash -l
#SBATCH -J=lisu                  # Job name  
#SBATCH -p=<partition>           # Specify cluster partition  
#SBATCH -N 2                     # Number of nodes  
#SBATCH --ntasks-per-node=4      # Tasks (processes) per node  
#SBATCH --gres=gpu:4             # GPUs per node  

srun python main.py fit -c configs/lisu_ptv3_l1+gtv+tgtv+eikonal_fp16_fulltrainval.yaml \
    --trainer.num_nodes 2 \      # Match SLURM node count  
    --trainer.devices 4          # GPUs per node  
```

### Self-Supervised Domain Adaptation
*(Coming soon: Instructions for adapting to real-world datasets like Waymo.)*

## TODOs <a name="todos"/>
- [x] Release dataset
  - [x] Release CARLA surface normal LiDAR sensor
- [x] Supervised training on LiSu dataset
- [ ] LiSu model checkpoints & evaluation
- [ ] Self-supervised domain adaptation for Waymo
- [ ] Waymo qualitative evaluation


## Citation <a name="citation"/>

```bibtex
@inproceedings{cvpr2025lisu,
  title={{LiSu: A Dataset and Method for LiDAR Surface Normal Estimation}},
  author={Du\v{s}an Mali\'c and Christian Fruhwirth-Reisinger and Samuel Schulter and Horst Possegger},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
} 
```
