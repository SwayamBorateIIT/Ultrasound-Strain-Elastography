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

This system was developed and tested on **in vivo ultrasound datasets**, with a focus on real-world medical imaging use cases. It can serve as a foundation for further exploration in elastography, medical diagnostics, or scientific imaging workflows.

