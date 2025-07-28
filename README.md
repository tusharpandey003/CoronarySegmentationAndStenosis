Coronary Segmentation and Stenosis Detection
This repository contains code, datasets, and experiments for a comprehensive project focused on coronary artery segmentation and stenosis detection using deep learning. It includes multiple datasets, segmentation and detection models, and experiment scripts with evaluation and visualization.

Overview
This project aims to develop and evaluate deep learning models for accurate segmentation of coronary arteries and classification of stenosis severity from multi-modal angiographic images. The work is organized into three main dataset domains and multiple modeling approaches:

ImageCAS: A large CTA image dataset with a state-of-the-art Swin Transformer model achieving superior Dice scores.

CADICA: Coronary stenosis detection models for severity classification into mild, moderate, and severe categories using Faster R-CNN and ResNet architectures.

Arcade: Segmentation of stenosis regions using UNet++ architecture on the Arcade dataset.

Each dataset folder contains scripts related to its data processing, model training, and evaluation.

Repository Structure
imagecas/
Contains deep learning models based on modern Transformer architectures (e.g., Swin Transformer).

swin.py – Implementation of Swin Transformer for coronary artery segmentation with improved Dice scores surpassing existing literature.

cadica/
Focused on stenosis detection from coronary angiograms with several model variants.

detect.py – Detection model classifying stenosis severity into mild, moderate, and severe classes, reporting accuracy metrics.

fasterrcnn.ipynb – Best performing detection model using Faster R-CNN architecture with visualization of predictions for robust validation.

exp-3.py – Experimental script using ResNet backbone for multi-class severity tasks (5 and 7 classes), providing benchmarks albeit not the final best metrics.

arcade/
Focuses on stenosis segmentation using UNet++ architecture.

Contains a notebook demonstrating training and evaluation strictly on the Arcade dataset.
