# Ensemble-Powered Deep Learning for Rice Leaf Disease Diagnosis

&#x20;

## Project Description

This repository contains a Jupyter Notebook that implements an end-to-end deep learning pipeline for **classifying rice leaf diseases** into 15 distinct categories. Leveraging PyTorch and torchvision, the notebook covers data loading, augmentation, model definition, training, and evaluation. Accurate disease detection on rice leaves can assist agronomists and farmers in early diagnosis, improving crop yields and reducing economic losses.

## Features

* **Data Loading & Exploration**: Load and inspect the rice leaf image dataset using the Kaggle API (`kagglehub`).
* **Data Augmentation**: Apply randomized transformations (rotations, flips, color jitter) to enhance model robustness.
* **Model Architecture**: Define and customize pre-trained CNN backbones (e.g., ResNet) for classification.
* **Training Loop**: Implement a modular training loop with configurable hyperparameters and real-time progress bars.
* **Evaluation & Reporting**: Compute accuracy, loss curves, confusion matrix, and detailed classification reports.
* **End-to-End Script**: Orchestrate the entire pipeline—from data download to model evaluation—with a single script.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/rice-leaf-disease-classification.git
   cd rice-leaf-disease-classification
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain Kaggle API credentials**:

   * Place `kaggle.json` in `~/.kaggle/`.

## Usage

1. **Launch Jupyter Lab**:

   ```bash
   jupyter lab
   ```

2. **Open the notebook**:

   * `Rice_Leafs_Disease_15.ipynb` in the root directory.

3. **Run cells sequentially**:

   * Start with Module 1 (Setup & Utility Functions) and follow through to End-to-End Script.

4. **Example: Reproduce Training Plot**:

   ```python
   from notebook_utils import plot_metrics

   history = torch.load('checkpoints/history.pt')
   plot_metrics(history, metrics=['loss', 'accuracy'])
   ```

## Notebook Walkthrough

### 1. Setup & Utility Functions

* Configure device (CPU/GPU), seed for reproducibility.
* Define helper functions: `set_seed`, `save_checkpoint`, metric calculators.

### 2. Data Loading

* Download dataset via Kaggle API using `kagglehub`.
* Create `ImageFolder` datasets and `DataLoader` for training and validation.

### 3. Data Augmentation & Transforms

* Define `torchvision.transforms` pipeline:

  * `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `Normalize`.

### 4. Model Definition

* Load a pre-trained ResNet backbone from `torchvision.models`.
* Replace the final fully connected layer for 15 classes.

### 5. Training Loop

* Train the model for configurable epochs.
* Log training/validation loss and accuracy.
* Utilize `tqdm` for progress.

### 6. Evaluation Metrics & Reports

* Plot loss/accuracy curves.
* Compute and display a confusion matrix and classification report using `sklearn.metrics`.

### 7. End-to-End Orchestration Script

* Combine all steps into a single Python script (`run_pipeline.py`) for automated execution.

## Dependencies

| Package      | Version |
| ------------ | ------- |
| Python       | 3.8+    |
| torch        | 1.13.0  |
| torchvision  | 0.14.0  |
| numpy        | 1.22.0  |
| matplotlib   | 3.5.0   |
| seaborn      | 0.11.2  |
| tqdm         | 4.64.0  |
| kagglehub    | 0.1.0   |
| scikit-learn | 1.0.2   |

## Results & Visuals

&#x20;*Figure 1: Training and validation loss/accuracy over epochs.*

&#x20;*Figure 2: Confusion matrix for test set predictions.*

## Contributing

We welcome contributions! Please:

1. **Fork** the repository.
2. **Create a new branch**: `git checkout -b feature/YourFeatureName`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature/YourFeatureName`
5. **Open a Pull Request**.

Feel free to open issues for bug reports or feature requests.

