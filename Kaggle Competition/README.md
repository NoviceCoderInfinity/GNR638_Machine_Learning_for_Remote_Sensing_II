# Kaggle Competition: Image Super Resolution Task on Custom Dataset containing Gaming Images

## Problem Statement
The objective of this assignment is to perform image super-resolution on a custom dataset of gaming images, where each low-resolution image has a corresponding high-resolution ground truth. The goal is to evaluate the performance and effectiveness of a Super-Resolution model, such as a Vision Transformer (ViT) or Convolutional Neural Network (CNN)-based approach, in reconstructing high-quality images from low-resolution inputs. The experiment aims to assess the model’s ability to enhance details, textures, and overall visual fidelity in gaming imagery.

## Overall Standings
Ranked 4 out of 61 teams

## Performance of Different Super-Resolution Models (480x270 → 1920x1080)

| **Model Description**                         | **Epochs** | **Score (PSNR)** |
|----------------------------------------------|------------|------------------|
| NLSA                                          | 100        | 57.9397          |
| **\*EDSR++**                                  | 200        | **58.5493**      |
| RCAN                                          | 50         | 57.5712          |
| **\*EDSR**                                    | 50         | **58.5604**      |
| SwinIR (pre-trained classical dataset)        | -          | 54.7761          |
| SRCNN with more epochs                        | 100        | 56.6312          |
| ViT                                           | 50         | 49.6894          |
| SRCNN with RGB Channels                       | 20         | 56.5044          |
| SRCNN approach                                | 20         | 50.8898          |

## Running the Code
All the code can be found in Best_Models and All_Models folder.
```plaintext
Best_Models
|---EDSR.py
|---EDSR_plus_plus.py

All_Models
|---NLSA.py
|---RCAN.py
|---ViTSR.py
|---SRCNN.py
```
