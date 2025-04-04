# Peasant-Identity: CNN-Based Farmer Recognition

This repository presents a deep learning pipeline for classifying individual farmers from image data using a custom Convolutional Neural Network (CNN). The project was developed for a vision-based identity recognition challenge aimed at improving agricultural workforce tracking and automation.

## Overview

The goal is to automatically identify different farmers using image classification techniques. The model learns to associate visual features with individual identities, enabling efficient logging of field activity and reducing reliance on manual supervision or video storage.

## CNN Model Structure

The model is a lightweight CNN designed for efficiency and generalization. It includes:

- **Three convolutional blocks**, each followed by batch normalization, ReLU activation, and max pooling
- **Adaptive average pooling** to flatten features from variable-sized inputs
- **A fully connected layer** that outputs the predicted identity class

This custom architecture is optimized for relatively small datasets but can be easily extended by incorporating pretrained models like ResNet or MobileNet for improved performance.

## Repository Structure

- `dataset/`: Input images for training, validation, and testing  
- `src/`: Core implementation  
  - `model.py`: CNN architecture  
  - `train.py`: Training script  
  - `val.py`: Evaluation logic  
  - `dataset.py`: Data loading and preprocessing  
- `tools/`: Utilities and helper functions  
- `result/test_csv/`: Model output predictions  
- `requirements.txt`: List of required Python packages
