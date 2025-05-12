
# GEC: Extinction Coordination for Enhanced 3D Gaussian Splatting

This repository is the official implementation of [GEC: Extinction Coordination for Enhanced 3D Gaussian Splatting]. 

This is a preview for review purposes only.

We are currently tidying up the code and will open‑source it once the paper is accepted.

## Requirements

To install requirements:

```setup
cd submodules
unzip diff-gaussian-rasterization.zip
unzip diff-gaussian-rasterization_analyse.zip
unzip fused-ssim.zip
unzip simple-knn.zip
cd ..
conda env create --file environment.yml
```

## Training

To train the model(s) in the paper, run this command:

```train
conda activate 3dgs_analyse
python train.py -s data/to/path/ -m path/to/save/ -r 2
```


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py -s data/to/path/ -m path/to/trained/
```


## Pre-trained Models

Our pretrained model could be found in supplementary materials.


