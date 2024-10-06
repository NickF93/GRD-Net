# GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module

This repository contains the **official implementation** of **GRD-Net**, a deep learning model designed for anomaly detection, specifically aimed at **visual inspection** tasks in industrial settings. Its goal is to identify defects by analyzing regions of interest (ROIs) within images, making it particularly useful for applications where only specific areas of a product may show relevant anomalies. The architecture comprises two main components:

## Model Architecture

1. **Generative Block**: This block is based on a **Generative Adversarial Network (GAN)** with a **residual autoencoder (ResAE)** architecture. The purpose of this component is to perform image reconstruction and denoising, generating clean versions of potentially defective images for further analysis.
   
2. **Discriminative Block**: This block is responsible for **defect localization** and anomaly detection. It analyzes the reconstructed images to spot defects, specifically focusing on predefined **Regions of Interest (ROIs)** where anomalies are likely to occur. The attention mechanism helps the network focus on the relevant areas of the image, reducing the computational cost and improving the precision of defect detection.

## Key Features

- **ROI Attention Module**: Unlike traditional methods that analyze the entire image, GRD-Net uses a Region of Interest (ROI) Attention Module, enabling the network to concentrate on specific areas where anomalies are most likely to appear. This reduces the reliance on post-processing methods such as blob analysis or image editing, which can be dataset-specific and hard to generalize.
  
- **Synthetic Data and Training**: The network is trained on a dataset composed of normal products and synthetically generated defects. This approach helps the model generalize well across different datasets, as it learns to identify defects without being biased by specific real-world defect patterns.

- **Reduction of Pre-processing**: By focusing on ROIs and leveraging synthetic data, GRD-Net eliminates the need for complex pre-processing algorithms, traditionally employed to spot defects in industrial settings. This leads to a more efficient pipeline and better generalization across diverse datasets.

## Datasets and Evaluation

GRD-Net has been evaluated on two main datasets:

- The **MVTec Anomaly Detection Dataset**, a benchmark dataset widely used for evaluating anomaly detection algorithms.
- A large-scale industrial dataset of **pharmaceutical BFS strips of vials**, representing a real-world use case where the method has shown significant performance improvements over traditional approaches.

## Why GRD-Net?

GRD-Net is designed to handle complex industrial inspection tasks where only specific regions of the product are of interest for defect detection. Traditional methods often struggle with generalization across datasets, requiring substantial pre-processing and post-processing. GRD-Net mitigates these issues by using its ROI-based attention mechanism and synthetic data for training.

## Keywords

- Anomaly Detection
- Attention Module
- Generative Adversarial Network (GAN)
- Defect Localization
- Region of Interest (ROI)

## Citation

To cite GRD-Net in your research, please use the following BibTeX entry:

```bibtex
@article{GRDNet-Ferrari-et-al-2023,
	title={GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module},
	author={Ferrari, Niccolò and Fraccaroli, Michele and Lamma, Evelina},
	journal={International Journal of Intelligent Systems},
	publisher={Wiley},
	year={2023},
	doi={10.1002/int.7773481}
	url={https://onlinelibrary.wiley.com/journal/1098111x}
}
