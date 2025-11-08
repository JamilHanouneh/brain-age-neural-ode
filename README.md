# Brain Aging Neural ODE

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-1806.07522-b31b1b.svg)](https://arxiv.org/abs/1806.07522)

A deep learning framework for predicting brain age from structural MRI using Neural Ordinary Differential Equations (Neural ODE) and normalizing flows. This implementation models brain aging as a continuous dynamical system, enabling interpretable age-specific brain templates and uncertainty quantification.

## Author

**Jamil Hanouneh**  
Master's in Medical Image and Data Processing  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

- Email: Jamil.hanouneh1997@gmail.com
- LinkedIn: [linkedin.com/in/jamil-hanouneh-39922b1b2](https://www.linkedin.com/in/jamil-hanouneh-39922b1b2/)
- GitHub: [github.com/JamilHanouneh](https://github.com/JamilHanouneh)

## Features

- Neural ODE-based continuous brain aging model
- Normalizing flow for latent space regularization
- PCA-based dimensionality reduction for efficient training
- Age-stratified evaluation metrics
- Brain age gap estimation with uncertainty quantification
- Generation of age-specific brain templates
- Comprehensive preprocessing pipeline for structural T1-weighted MRI
- TensorBoard integration for training visualization

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Generate Templates](#4-generate-templates)
- [Configuration](#configuration)
- [Understanding the Results](#understanding-the-results)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

1. Clone the repository:
```
git clone https://github.com/JamilHanouneh/brain-aging-neural-ode.git
cd brain-aging-neural-ode
```

2. Create a virtual environment:
```
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Setup (Optional)

For NVIDIA GPU support:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

This project uses the **IXI Dataset** (Information eXtraction from Images), a publicly available dataset of brain MRI scans.

### Download IXI Dataset

1. Visit the [IXI Dataset website](https://brain-development.org/ixi-dataset/)

2. Download T1-weighted structural MRI scans (requires registration)

3. Download the metadata file (IXI.xls)

4. Organize files as follows:
```
data/raw/IXI/
├── IXI002-Guys-0828-T1.nii.gz
├── IXI012-HH-1211-T1.nii.gz
├── IXI013-HH-1212-T1.nii.gz
├── ...
└── IXI.xls
```

Expected: ~581 NIFTI files (`.nii.gz` format) + 1 Excel metadata file

### Dataset Statistics

- Total subjects: 581
- Age range: 20-86 years
- Modality: T1-weighted MRI
- Scanner: 1.5T and 3T
- Hospitals: Guys Hospital, Hammersmith Hospital, Institute of Psychiatry

## Project Structure

```
brain-aging-neural-ode/
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Data directory
│   ├── raw/IXI/                # Raw IXI dataset
│   ├── processed/              # Preprocessed data
│   └── metadata/               # Metadata files
├── scripts/                     # Executable scripts
│   ├── preprocess_data.py      # Data preprocessing
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   └── generate_templates.py   # Template generation
├── src/                         # Source code
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training utilities
│   ├── inference/              # Inference utilities
│   ├── registration/           # Image registration
│   └── utils/                  # Helper functions
├── outputs/                     # Training outputs
│   ├── models/                 # Saved models
│   ├── logs/                   # Training logs
│   └── evaluation/             # Evaluation results
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Usage

### 1. Data Preprocessing

Preprocess the IXI dataset:

```
python scripts/preprocess_data.py --config config/config.yaml --dataset IXI
```

This script:
- Loads raw NIFTI files
- Applies skull stripping
- Normalizes intensity values (Z-score)
- Resamples to standard shape (96 x 112 x 96)
- Splits data into train/val/test (70%/15%/15%)
- Applies PCA dimensionality reduction (1M voxels -> 393 components)
- Saves preprocessed data to `data/processed/`

Expected runtime: ~30 minutes (CPU)

### 2. Training

Train the brain aging model:

```
python scripts/train.py --config config/config.yaml
```

Training options:
- **Resume from checkpoint:**
  ```
  python scripts/train.py --config config/config.yaml --resume outputs/models/checkpoint_epoch_50.pt
  ```

- **Monitor with TensorBoard:**
  ```
  tensorboard --logdir outputs/logs/tensorboard
  ```
  Open browser to `http://localhost:6006`

Expected runtime:
- CPU: 12-24 hours for 100 epochs
- GPU: 2-4 hours for 100 epochs

Training output:
```
Epoch 1/100
  Train Loss: 886.57, Age Loss: 468.24
  Val Loss: 1642.65, MAE: 31.29 years
  
Epoch 50/100
  Train Loss: 234.12, Age Loss: 89.45
  Val Loss: 432.76, MAE: 8.43 years
  Best model saved!
```

### 3. Evaluation

Evaluate the trained model on the test set:

```
python scripts/evaluate.py \
    --config config/config.yaml \
    --checkpoint outputs/models/best_model.pt \
    --output-dir outputs/evaluation
```

This generates:
- **Metrics** (JSON): MAE, RMSE, R², Pearson/Spearman correlations
- **Plots** (PNG):
  - Age prediction scatter plot
  - Brain age gap distribution
  - Residual plot
  - Bland-Altman plot
- **Age-stratified analysis**: Performance across different age groups

### 4. Generate Templates

Generate age-specific brain templates:

```
python scripts/generate_templates.py \
    --config config/config.yaml \
    --checkpoint outputs/models/best_model.pt \
    --ages 30 40 50 60 70 80 \
    --output-dir outputs/templates
```

This creates:
- NIFTI files for each age: `template_age_30.nii.gz`, etc.
- Visualization comparing templates across ages

## Configuration

Edit `config/config.yaml` to modify hyperparameters:

### Key Parameters

```
# Model architecture
model:
  input_dim: 393              # PCA components
  latent_dim: 393             # Latent space dimension
  hidden_dim: 512             # Hidden layer size
  
  neural_ode:
    solver: 'dopri5'          # ODE solver
    rtol: 0.001               # Relative tolerance
    atol: 0.0001              # Absolute tolerance

# Training parameters
training:
  batch_size: 8               # Samples per batch
  num_epochs: 100             # Total epochs
  learning_rate: 0.0001       # Learning rate
  optimizer: 'adam'           # Optimizer
  scheduler: 'cosine'         # LR scheduler
  
  early_stopping:
    enabled: true
    patience: 15              # Epochs before stopping
    
  loss_weights:
    age_prediction: 1.0       # Age prediction weight
    reconstruction: 0.5       # Reconstruction weight
    latent_prior: 0.1         # Latent prior weight

# Data preprocessing
data:
  preprocessing:
    target_shape:   # Target dimensions
    normalize: true               # Z-score normalization
    pca_components: 500           # PCA components (max)
```

### Hyperparameter Tuning Tips

1. **Fast testing** (CPU, quick results):
   ```
   training:
     batch_size: 4
     num_epochs: 10
     early_stopping:
       patience: 3
   ```

2. **Production** (GPU, best results):
   ```
   training:
     batch_size: 32
     num_epochs: 200
     learning_rate: 0.0005
   ```

3. **Increase model capacity**:
   ```
   model:
     hidden_dim: 1024
     num_layers: 4
   ```

## Understanding the Results

### Metrics Explained

- **MAE (Mean Absolute Error)**: Average age prediction error in years
  - Good: < 5 years
  - Acceptable: 5-10 years
  - Poor: > 10 years

- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
  - Similar to MAE but higher if predictions vary widely

- **R² (Coefficient of Determination)**: How well predictions fit true ages
  - 1.0 = perfect fit
  - 0.0 = no better than mean prediction
  - < 0 = worse than mean (poor model)

- **Pearson r**: Linear correlation between predicted and true age
  - Good: > 0.85
  - Acceptable: 0.70-0.85
  - Poor: < 0.70

- **Brain Age Gap**: Difference between predicted and chronological age
  - Positive = brain appears older (accelerated aging)
  - Negative = brain appears younger (preserved aging)
  - Clinical relevance for neurodegenerative diseases

### Age-Stratified Analysis

Performance often varies by age group:
```
Age 20-30: MAE = 4.2 years  (Good)
Age 30-40: MAE = 6.5 years  (OK)
Age 40-50: MAE = 13.3 years (Getting worse)
Age 60+:   MAE = 30+ years  (Poor)
```

This indicates the model struggles with older brains, possibly due to:
- Fewer older subjects in training data
- Greater biological variability in aging
- Need for more data augmentation

### Interpreting Plots

1. **Scatter plot**: Points should lie along diagonal (y=x)
2. **Residual plot**: Random scatter around zero (no systematic bias)
3. **Bland-Altman**: 95% of points within ±1.96 SD
4. **Brain age gap**: Should be normally distributed around zero

## Model Architecture

### Neural ODE Flow

```
Input (PCA-reduced brain)
    |
    v
[Encoder] (393 -> 393)
    |
    v
[Neural ODE] (continuous dynamics)
    dh/dt = f(t, h)  <-- Learned dynamics function
    Solve ODE from t=0 to t=1
    |
    v
[Age Predictor] (393 -> 1)
    |
    v
Output (predicted age)
```

### Key Components

1. **Encoder**: Compresses brain features to latent representation
2. **Neural ODE**: Models aging as continuous transformation
3. **ODE Solver**: Uses Dormand-Prince (dopri5) for accurate integration
4. **Age Predictor**: Maps latent state to predicted age

### Loss Function

```
L_total = λ₁·L_age + λ₂·L_recon + λ₃·L_prior

where:
- L_age: MSE between predicted and true age
- L_recon: Reconstruction loss (autoencoder)
- L_prior: KL divergence (latent regularization)
```

## Citation

If you use this code in your research, please cite:

```
@mastersthesis{hanouneh2025brainaging,
  author  = {Jamil Hanouneh},
  title   = {Brain Aging Prediction Using Neural Ordinary Differential Equations},
  school  = {Friedrich-Alexander-Universität Erlangen-Nürnberg},
  year    = {2025},
  type    = {Master's Thesis},
  url     = {https://github.com/JamilHanouneh/brain-aging-neural-ode}
}
```

And the original Neural ODE paper:
```
@inproceedings{chen2018neuralode,
  title     = {Neural Ordinary Differential Equations},
  author    = {Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IXI Dataset: [brain-development.org](https://brain-development.org/ixi-dataset/)
- Neural ODE implementation: [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- Brain preprocessing tools: [NiBabel](https://nipy.org/nibabel/), [SimpleITK](https://simpleitk.org/)

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: Jamil.hanouneh1997@gmail.com
- LinkedIn: [Jamil Hanouneh](https://www.linkedin.com/in/jamil-hanouneh-39922b1b2/)

---

Made with ❤️ by Jamil Hanouneh at FAU Erlangen-Nürnberg
```
