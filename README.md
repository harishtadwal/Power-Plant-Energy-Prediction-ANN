# Power Plant Energy Prediction with an Artificial Neural Network

This project uses a PyTorch artificial neural network (ANN) to predict the
electrical energy output of a combined-cycle power plant from four ambient and
operating measurements.

## Project overview

The model performs supervised regression. It learns the relationship between:

- **AT**: Ambient Temperature
- **V**: Exhaust Vacuum
- **AP**: Ambient Pressure
- **RH**: Relative Humidity

and the target:

- **PE**: Net Electrical Energy Output

The dataset contains **9,568 observations** and has no missing values. The
notebook uses an 80/20 train-test split with `random_state=42`.

## Model architecture

The ANN is implemented in PyTorch with:

- Input layer: 4 features
- Hidden layer 1: 6 neurons with ReLU activation
- Hidden layer 2: 6 neurons with ReLU activation
- Output layer: 1 neuron for regression
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 32
- Training duration: 100 epochs

The input features are standardized using `StandardScaler`, fitted only on the
training data.

## Results

The saved model was evaluated on the held-out test set. The results recorded in
[`ann.ipynb`](./ann.ipynb) are:

| Metric | Value |
|---|---:|
| Test MSE | 19.5328 |
| Test R² | 0.9317 |

The trained model weights are stored in [`best_model.pt`](./best_model.pt).

## Repository structure

```text
.
├── ann.ipynb          # Data preparation, training, evaluation, and plots
├── best_model.pt      # Saved PyTorch state dictionary
├── powerplant.csv     # Dataset
└── README.md          # Project documentation
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/harishtadwal/Power-Plant-Energy-Prediction-ANN.git
cd Power-Plant-Energy-Prediction-ANN
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib torch jupyter
```

### 4. Run the notebook

```bash
jupyter notebook ann.ipynb
```

Run the cells in order to:

1. Load and inspect the dataset.
2. Split and standardize the data.
3. Build and train the ANN.
4. Save the best model to `best_model.pt`.
5. Plot training and validation loss.
6. Evaluate the model using MSE and R².

## Loading the trained model

The model weights are saved as a PyTorch `state_dict`. To load them, first
define the same `ANN` class from the notebook:

```python
import torch

model = ANN()
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
```

Before making a prediction, apply the same `StandardScaler` preprocessing used
during training. A prediction must contain the features in this order:

```text
AT, V, AP, RH
```

## Dataset

The columns and measurements are based on the Combined Cycle Power Plant
dataset. The local copy used by this project is available in
[`powerplant.csv`](./powerplant.csv).

## Notes

- `best_model.pt` contains model weights, not the complete model definition or
  the fitted scaler.
- For production use, save the fitted scaler alongside the model and validate
  predictions on data representative of the deployment environment.
- This repository is intended for educational and portfolio purposes.

