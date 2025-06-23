# Ultrasound-Strain-Elastography

## POSTER 

![image](https://github.com/user-attachments/assets/5f4159a1-dc01-4f76-aca7-2e440d4494f8)

## 🩺 Project Overview: Ultrasound Strain Elastography

This project implements a complete pipeline for **strain elastography using ultrasound imaging**, a non-invasive technique to estimate and visualize the mechanical properties (like stiffness) of soft tissues. By analyzing tissue displacement before and after external compression, the method produces a strain map that highlights variations in elasticity — often useful in detecting abnormalities such as tumors or lesions.

The core of this project involves using a deep neural network to learn a mapping from 2D spatial coordinates to pixel intensities, effectively reconstructing ultrasound strain images. The pipeline includes:

- 📥 **Data Preprocessing**: Load and prepare ultrasound sequences.
- 🧠 **Model Training (PyTorch)**: Learn coordinate-to-pixel mappings using supervised learning.
- 🖼️ **Image Reconstruction**: Generate strain images from predicted output.
- 🛑 **Early Stopping**: Automatically stop training when the model converges.

This system was developed and tested on **in vivo ultrasound datasets** sourced from the **IMPACT Laboratory** at Concordia University. The data (2200 RF data pairs acquired with a CIRS phantom and Alpinion E-Cube R12 machine) originates from their “Ultrasound Elastography Dataset for Unsupervised Training” :contentReference[oaicite:1]{index=1}.

---

## 🧠 Methodology: Integrating DIC Principles into Ultrasound Elastography

### Why Digital Image Correlation (DIC)?

Digital Image Correlation is a well-established, non-contact optical method used in experimental mechanics to compute full-field displacements and strains on surfaces. It is highly effective in textured image fields, where patterns allow for pixel-wise tracking under deformation.

In our approach, we leverage the **natural speckle patterns** present in **B-mode ultrasound images** as the texture necessary for DIC-style tracking. Unlike traditional elastography methods that rely on RF data, our model operates on **B-mode data only**, which is:

- ✅ Universally available across scanners
- ✅ Easier to process and integrate into clinical workflows
- ✅ Lower in storage and computational demand

This B-mode compatibility makes our method highly practical for widespread clinical deployment and for building **ground truth-free, real-time strain mapping tools**.

---

## 🧩 Model Overview

Inspired by [DICNet-corr](https://github.com/fead1/DICNet-corr-unsupervised-learning-), our architecture consists of:

- A **Siamese encoder-decoder network** to extract features from pre- and post-compression B-mode images.
- A **correlation layer** to measure similarity and track displacements.
- A **warp and update module** to iteratively refine estimated displacement fields.

This design enables our model to learn **unsupervised strain estimation** purely from deformation-consistent patterns in ultrasound image sequences.

---

## 🧮 Loss Functions

We implement a combination of pixel- and patch-based unsupervised loss functions that do not require ground truth displacement or strain:

### 1. Patch-ZNSSD Loss
A patch-based normalized sum of squared differences, robust to illumination variations.

$
L_{\text{ZNSSD}} = \sum \left( \frac{(f - \mu_f)}{\sigma_f} - \frac{(g - \mu_g)}{\sigma_g} \right)^2
$

### 2. Smoothness Loss
Encourages spatial smoothness in the predicted displacement field.

$
L_{\text{smooth}} = \sum ||\nabla u||^2 + ||\nabla v||^2
$

### 3. Census Loss
A robust loss that compares pixel neighborhoods instead of absolute values.

$
L_{\text{census}} = \sum |C(f) - C(g)|
$

The total training loss is a weighted combination of these three losses:

$
L = \lambda_1 L_{\text{ZNSSD}} + \lambda_2 L_{\text{smooth}} + \lambda_3 L_{\text{census}}
$

---

## 📊 Evaluation & Metrics

Our model was evaluated on:

- 🧪 **Synthetic & Phantom Datasets** (Alpinion and ABAQUS)
- 🩺 **In-vivo Data** (clinical ultrasound frames with no ground truth)

We compare our results to **GLUE (GLocal Ultrasound Elastography)**, a traditional RF-based algorithm.

### Metrics:

- **NRMSE (%):** Normalized Root Mean Squared Error
$
\text{NRMSE}(\%) = \left( \frac{100}{z} \sqrt{\frac{1}{N} \sum (x_i - x_i^*)^2} \right)
$

- **SNRe (dB):** Signal-to-Noise Ratio on estimated strain maps
$
\text{SNRe} = \frac{\mu_s}{\sigma_s}
$

---

## ✅ Highlights

- 🌟 **Ground Truth-Free Training** using unsupervised losses.
- 🌟 **B-Mode-Only Methodology** for broad applicability.
- 🌟 **Comparable to RF-based GLUE** for small and medium strains.
- 🌟 **Handles in-vivo scenarios** with moderate success (room for generalization improvements).
- 🌟 **Compact model** that can be adapted for real-time strain visualization in portable ultrasound devices.

---





## 📚 References

### Dataset
Our ultrasound data was sourced from the **IMPACT Laboratory at Concordia University**, specifically the *“Ultrasound Elastography Dataset for Unsupervised Training”* :contentReference[oaicite:1]{index=1}.

### Model
The core neural network architecture in this project is adapted from the **DICNet-corr unsupervised learning framework** :contentReference[oaicite:2]{index=2}, originally developed by `fead1`. DICNet‑corr is an unsupervised CNN-based method for 2D displacement measurement using digital image correlation.



