# Hopfield Network MNIST Demo

This project demonstrates a Hopfield Network for associative memory using the MNIST handwritten digit dataset.  
It trains the network to remember selected digits and reconstructs them from noisy/corrupted inputs.

## Features

- Train Hopfield Network on MNIST digits (configurable, e.g. 7, 8, 9)
- Visualize original, corrupted, and reconstructed images
- Save and load trained weights for consistent results

## Usage

1. Install dependencies:
    ```bash
    pip install numpy matplotlib scikit-image tqdm keras
    ```

2. Run training and prediction:
    ```bash
    python train_predict_mnist.py
    ```

3. To reuse good weights:
    - After a good training run, weights are saved.
    - On later runs, load weights and skip training.

## Files

- `train_predict_mnist.py` — Main script for training and testing
- `hopfield_network.py` — Hopfield Network implementation
- `.gitignore` — Ignore unnecessary files
- `README.md` — Project info
