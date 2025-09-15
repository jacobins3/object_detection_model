# Settlement and Farmland Detection Models

This repository hosts trained deep learning models for settlement detection and farmland classification using high-resolution satellite imagery.

## Models

Mask R-CNN (Settlements)

Task: Detect and segment built-up/settlement areas.

Training settings: Learning rate = 0.0002, Batch size = 12.

SAM-LoRA (Farmlands)

Task: Classify and segment agricultural land.

Training settings: Learning rate = 0.001, Batch size = 12.

## Data

The models were trained on annotated high-resolution satellite imagery from a critical protected area to analyze agricultural and settlement expansion.

Usage

Clone the repository

git clone https://github.com/jacobins3/object_detection_model.git
cd settlement-farmland-models-BMNP


Load the trained model weights into your preferred deep learning framework (e.g., PyTorch).

Run inference on satellite imagery to extract settlement and farmland masks.

## Requirements

Python ≥ 3.9

PyTorch ≥ 2.0

torchvision

segment-anything (for SAM-LoRA)

detectron2 (for Mask R-CNN)

## Citation

If you use these models in your work, please cite this repository.
