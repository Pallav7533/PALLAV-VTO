# Virtual Try-On Application with CAT-VTON and IDM-VTON

This project implements a virtual try-on system using two state-of-the-art models, **CAT-VTON** and **IDM-VTON**. The system allows users to visualize clothing on models, enabling a realistic virtual try-on experience. The dataset used is **DeepFashion** from Hugging Face, containing images of fashion models and apparel items.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)


---

## Project Overview

The goal of this project is to create a virtual try-on application that can overlay clothing items onto model images realistically. Using **CAT-VTON** and **IDM-VTON** models, this application:
1. Processes input images to align clothing with model images.
2. Applies the clothing item onto the model image, creating a virtual try-on effect.

## Dataset

We use the **DeepFashion** dataset from Hugging Face, which provides a rich set of images for virtual try-on applications.

To load the dataset:

from datasets import load_dataset
ds = load_dataset("lirus18/deepfashion")

## Requirements

Python 3.7+
PyTorch
CUDA (if using GPU for faster inference)
Additional libraries: torchvision, datasets, numpy, opencv-python

!pip install torch torchvision datasets numpy opencv-python

## Setup

1. Clone the Repositories
Clone the required repositories for CAT-VTON and IDM-VTON models:

git clone https://github.com/username/CAT-VTON.git
git clone https://github.com/username/IDM-VTON.git

2. Download Pre-trained Checkpoints
3. 3. Data Preprocessing
Prepare the dataset images according to the model requirements:

Resize images to the required input size (256x192 for most virtual try-on models).
Generate pose estimations or segmentation maps if required by the model.

## Usage

Running Inference
After setup, follow these steps to run the virtual try-on process:

Load Sample Images: Place images in a folder (data/input) for models and clothing items.

Run CAT-VTON or IDM-VTON Inference: Use the respective scripts in each repository to execute the try-on.

Example command for CAT-VTON:   python cat_vton_inference.py --input_dir data/input --output_dir results/cat_vton_output

Example command for IDM-VTON:  python idm_vton_inference.py --input_dir data/input --output_dir results/idm_vton_output


View Results: Output images will be saved in results/cat_vton_output or results/idm_vton_output, showing models with applied clothing items.

## Results

Sample results are saved in the results folder. The application demonstrates realistic try-on functionality by accurately applying clothing items onto model images.


