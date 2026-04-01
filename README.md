# ML4SCI Gravitational Lensing Monorepo

Welcome to the unified repository for the Google Summer of Code (GSoC) ML4SCI project.
This repository serves as a centralized monolithic hub for multiple independent models and investigations developed to simulate and classify strong gravitational lensing systems. 

Each component is completely modular and contained within its respective subdirectory. You will find self-contained Jupyter Notebooks, metric reports, generated images, and detailed `README.md` documents tailored for each specific task in these folders.

## Project Structure

### 1. [Lensing Generation (DDPM & VAE)](./Lensing_Generation_DDPM/README.md)
This directory contains the original work focused on synthetically generating simulated strong gravitational lenses using a Variational Autoencoder (VAE). It efficiently generates 64x64 lens approximations and scores them against strict physical constraints (Radial Profile RMSE, Power Spectrum RMSE, and FID quality scores).
* **Notebook**: `VAE_DDPM.ipynb`
* **Artifacts**: `vae_outputs/` (grids, loss curves, metric reports)

### 2. [Lensing Classification (Physics-Informed Neural Networks) ](./Lensing_Classification_PINN/README.md)
This directory incorporates the robust PINN approach (Physics-Guided-ML) previously housed in a separate repository. This model injects physics domain constraints directly into the neural network architecture allowing the classifier to identify properties and lensing signatures with greater confidence and accuracy.
* **Notebook**: `pinn_lensing_classifier.ipynb` 
* **Artifacts**: Receiver Operating Characteristic (ROC) plots, training histories.

### 3. [Lensing Classification (Task 2 ResNet Base)](./Lensing_Classification_Task2/README.md)
This segment hosts the baseline Deep Learning classification model designed as part of the primary ML4SCI application tasks (`Task2_Classification`). It tests standard classification architectures mapping input observational data onto discrete theoretical structures. 
* **Notebook**: `Classification.ipynb`
* **Artifacts**: Detailed `results/` metrics evaluating classification precision.

---

*This unified monorepo setup ensures that all methodologies, experiments, and results remain cleanly distinct, making the codebase easier to review, execute, and iterate upon.*
