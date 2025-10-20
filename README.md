# FiberGNN: A Graph Neural Network Framework for Efficient Photonic Crystal Fiber Parameters Prediction

This repository contains the official code implementation for the paper: "FiberGNN: A Graph Neural Network Framework for Efficient Photonic Crystal Fiber Parameters Prediction".

## Overview

FiberGNN is a Graph Neural Network (GNN) framework designed to efficiently and accurately predict key optical parameters (e.g., effective refractive index, dispersion) of Photonic Crystal Fibers (PCFs) directly from their microstructures. By representing PCFs as graphs, FiberGNN overcomes the limitations of traditional simulation methods and other machine learning approaches, demonstrating strong generalization capabilities, especially for disordered structures.

This repository includes:
* The core FiberGNN model implementation (`models.py`, using PyTorch Geometric).
* The main script for training the model (`FiberGNN.py`).
* Supporting scripts for configuration (`configs.py`), data loading (`dataloader.py`), training (`trainer.py`), utilities (`utils.py`), and visualization (`visualizer.py`).
* The source code for the Flask-based online prediction platform (`Application/flask.py` and related files).
* Dependencies file (`requirements.txt`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YPL-97/FiberGNN.git](https://github.com/YPL-97/FiberGNN.git)
    cd FiberGNN
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *(Note: The file in the repository is named `requirement.txt`. Please ensure it's renamed to `requirements.txt` or update the command below.)*
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct version of PyTorch installed for your system (CPU or specific CUDA version). See the [PyTorch official website](https://pytorch.org/) for installation details.*

## Usage

### Training the Core Model

The main script for training the FiberGNN model is `FiberGNN.py`.

```bash
python FiberGNN.py --data_path ./path/to/your/dataset --epochs 50 --batch_size 32 --lr 1e-4 # Add other relevant arguments as defined in the script/configs.py
# --data_path: Specify the path to your training/validation data.
# --epochs, --batch_size, --lr: Example hyperparameters. Check configs.py or the script's argument parser for all options.

### Running the Online Platform (Flask Web Application)

The web application allows for interactive predictions using a pre-trained model.

1. Navigate to the application directory:
cd Application
2. Run the Flask application:
python flask.py
