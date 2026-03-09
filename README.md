# Deep-MC-QR

[![CI/CD Pipeline](https://github.com/Xiaohu-Zheng/Deep-MC-QR/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/Xiaohu-Zheng/Deep-MC-QR/actions)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Deep Monte Carlo Quantile Regression (Deep-MC-QR)** - Quantifying Aleatoric Uncertainty in Physics-informed Temperature Field Reconstruction

## 📖 Overview

This repository implements the Deep-MC-QR algorithm presented in the paper:

> **Deep Monte Carlo Quantile Regression for Quantifying Aleatoric Uncertainty in Physics-informed Temperature Field Reconstruction**  
> *Zheng, Xiaohu et al.*

## 🌟 Key Features

- **Uncertainty Quantification**: Quantifies aleatoric uncertainty using quantile regression
- **Monte Carlo Sampling**: Employs Monte Carlo methods for robust predictions
- **Deep Learning Integration**: Combines deep neural networks with quantile regression
- **Physics-informed**: Incorporates physical constraints in temperature field reconstruction
- **Flexible Architecture**: Supports various backbone networks (U-Net, VGG, ResNet, FPN)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Xiaohu-Zheng/Deep-MC-QR.git
cd Deep-MC-QR

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the main training script
python main.py
```

### Example: Temperature Field Reconstruction

```python
import torch
from src.DeepRegression import DeepRegressionModel

# Initialize model
model = DeepRegressionModel(
    backbone='unet',
    quantiles=[0.1, 0.5, 0.9]
)

# Train model
model.train(train_loader, val_loader, epochs=100)

# Predict with uncertainty quantification
predictions = model.predict(test_data)
```

## 📁 Project Structure

```
Deep-MC-QR/
├── src/                        # Source code
│   ├── models/                # Neural network architectures
│   │   ├── unet.py           # U-Net model
│   │   ├── fpn.py            # Feature Pyramid Network
│   │   └── backbone/         # Backbone networks
│   │       ├── vgg.py
│   │       ├── resnet.py
│   │       └── alexnet.py
│   ├── mcqr/                  # Monte Carlo Quantile Regression
│   │   └── mcqr_regression.py
│   ├── loss/                  # Loss functions
│   │   └── loss.py
│   ├── utils/                 # Utility functions
│   │   ├── model_init.py
│   │   └── np_transforms.py
│   ├── DeepRegression.py      # Main regression model
│   ├── train.py               # Training utilities
│   ├── data_noise.py          # Data augmentation
│   └── plot.py                # Visualization tools
├── config/                     # Configuration files
├── tests/                      # Test suite
│   └── test_basic.py
├── .github/workflows/          # CI/CD configuration
│   └── ci.yml
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/ --line-length 100

# Check code quality
flake8 src/ tests/ --max-line-length 100

# Sort imports
isort src/ tests/ --profile black
```

## 🎯 Algorithm Description

Deep-MC-QR combines deep learning with quantile regression for uncertainty quantification:

### Core Concepts

1. **Quantile Regression**: Predicts conditional quantiles instead of point estimates
2. **Monte Carlo Sampling**: Uses random sampling for uncertainty estimation
3. **Deep Neural Networks**: Learns complex nonlinear relationships
4. **Physics-informed Constraints**: Incorporates domain knowledge

### Mathematical Foundation

The quantile loss function:

```
L(y, ŷ_τ) = max(τ(y - ŷ_τ), (τ-1)(y - ŷ_τ))
```

where:
- `y` is the true value
- `ŷ_τ` is the predicted τ-th quantile
- `τ ∈ [0, 1]` is the quantile level

## 📊 Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **U-Net** | Encoder-decoder architecture | Image-to-image tasks |
| **FPN** | Feature Pyramid Network | Multi-scale feature extraction |
| **VGG** | VGG backbone | Transfer learning |
| **ResNet** | Residual networks | Deep architectures |
| **AlexNet** | Classic CNN | Baseline comparisons |

## 📈 Performance

- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Robust Predictions**: Handles noisy and incomplete data
- **Flexible Architecture**: Easy to adapt to different problems

## 🧪 Examples

### 1. Basic Training

```bash
python main.py --config config/default.yaml
```

### 2. Custom Quantiles

```python
from src.mcqr.mcqr_regression import MCQRRegression

model = MCQRRegression(
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)
```

### 3. Visualization

```python
from src.plot import plot_quantile_predictions

plot_quantile_predictions(
    true_values,
    predictions,
    quantiles=[0.1, 0.5, 0.9],
    save_path='results.png'
)
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{Zheng2023DeepMCQR,
   author = {Zheng, Xiaohu and others},
   title = {Deep Monte Carlo Quantile Regression for Quantifying Aleatoric Uncertainty in Physics-informed Temperature Field Reconstruction},
   journal = {Journal Name},
   year = {2023}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Xiaohu Zheng
- **Email**: zhengxiaohu16@nudt.edu.cn
- **GitHub**: [@Xiaohu-Zheng](https://github.com/Xiaohu-Zheng)

## 🙏 Acknowledgments

- National University of Defense Technology
- Contributors and collaborators

---

**Star ⭐ this repository if you find it helpful!**
