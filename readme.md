# Galaxy Morphology Classification with Deep Learning 🌌 

**Authors:** Owen BRAUX & Elliot CAMBIER  
**Context:** Deep learning 2 Project ESIEE Paris (2026)

## Goal
The goal of this project is to classify galaxies into **10 distinct morphological classes** (E0, Sb, Barred spirals, Mergers, etc.) using Deep Learning.

We used data from the **Galaxy Zoo 2** project, which is challenging because the images are noisy, low-resolution and the classes are visually very similar (Fine-Grained Classification).

## Project Structure

The project is divided into two main notebooks to separate Data Engineering from Machine Learning:

* **`01_Data_Pipeline.ipynb`** : 
    * Downloads the raw CSV catalog from Galaxy Zoo.
    * Converts soft votes into **Hard Labels** using a custom decision tree.
    * Balances the dataset.
    * Downloads images via the SDSS API.
    * **Output:** A clean `dataset_galaxies.zip` ready for training.

* **`02_model-training-***.ipynb`** :
    * Loads the data with PyTorch.
    * Applies Data Augmentation (Crucially: **Random Rotation 0-180°** because galaxies have no specific orientation).
    * Compares 3 different architectures.

## Approach & Models

We followed a scientific approach to compare different architectures, from simple to complex:

### 1. Baseline : Fully Connected
* **Architecture:** We flattened the images (64x64 pixels) into a vector and used dense layers.
* **Hypothesis:** This should fail because it destroys spatial structure.
* **Result:** ~10% Accuracy (Same as random guessing so = useless).

### 2. From Scratch : Custom CNN
* **Architecture:** A standard Convolutional Neural Network (3 Conv blocks + MaxPool).
* **Input:** 128x128 images.
* **Result:** ~31% Accuracy. It sees shapes but struggles with details.

### 3. Transfer Learning : MobileNetV2 (the best)
* **Architecture:** We used a pre-trained MobileNetV2.
* **Strategy:** We used a **"Two-Stage Training"**:
    1.  **Frozen Backbone:** Train only the classifier head to adapt to the new classes.
    2.  **Fine-Tuning:** Unfreeze the whole network with a very low Learning Rate (`1e-4`) to adapt the deep features to astronomy.
* **Result:** ~49% Accuracy.

## Results Summary

| Model | Type | Accuracy (Val) | Observation |
| :--- | :--- | :--- | :--- |
| **Simple FC** | MLP | ~10% | Fails to capture geometry |
| **Custom CNN** | From Scratch | ~31% | Good start |
| **MobileNetV2** | Transfer Learning | **~49%** | Best performance. |

> **Note on Accuracy:** While 49% might seem low compared to standard datasets (cats/dogs), it is a strong result for **10-class fine-grained galaxy classification**. Papers achieving 90%+ usually classify only 3 macro-classes (Spiral vs Elliptical vs Artifact). When grouping our classes into these 3 categories, our accuracy exceeds 90%.

## How to Run

1.  **Prepare Data:** Run `01_Data_Pipeline.ipynb`. It will create the folder structure and download images.
2.  **Train Models:** Run `02_model-training.ipynb`. It will train the model of your choice.

### Prerequisites
* Python 3.8+
* pandas>=1.3.0
* numpy>=1.21.0
* requests>=2.26.0
* Pillow>=8.3.0
* tqdm>=4.62.0
* torch>=1.9.0
* torchvision>=0.10.0
* matplotlib>=3.4.0
* scikit-learn>=1.8.0
* seaborn>=0.13.2