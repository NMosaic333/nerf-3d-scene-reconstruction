## Neural Radiance Field (NeRF) 3D Scene Reconstruction



Implementation of Neural Radiance Fields for neural volumetric rendering and novel view synthesis using TensorFlow.



## Overview



This project implements Neural Radiance Fields (NeRF) to reconstruct a 3D scene from multi-view images and generate novel viewpoints.



The model learns a continuous radiance field representation of the scene and uses volume rendering to synthesize new views.



## Inspiration



Inspired by the paper:



NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis



## Results



The NeRF model was evaluated on 160+ test views using standard neural rendering metrics.



| Metric | Value |

|------|------|

| PSNR | 24.8 dB |

| SSIM | 0.94 |

## Rendered Results

Ground Truth vs Rendered View

## Installation

git clone https://github.com/NMosaic333/nerf-3d-scene-reconstruction.git

cd nerf-3d-scene-reconstruction

pip install -r requirements.txt

## Configuration

Experiment parameters are controlled through `configs/nerf_config.yaml`,
allowing easy modification of training, rendering, and model settings.

## Dataset

This project uses the NeRF synthetic dataset from Kaggle.

Download it from:
https://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset

After downloading, extract the dataset into the following directory:

data/

## Training

python scripts/train.py

## Rendering

python scripts/render.py

## Evaluation

python scripts/evaluate.py

## References
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2021). Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1), 99-106.