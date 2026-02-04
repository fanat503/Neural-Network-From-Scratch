# Neural Network from Scratch (MNIST) 

A lightweight, 3-layer Multi-Layer Perceptron (MLP) implemented from the ground up using only **Python**, **NumPy**, and **SciPy**. This project was built to master the fundamental mathematics of Deep Learning without relying on high-level frameworks like TensorFlow or PyTorch.

## Key Features 
- **Architecture**: 3-layer feed-forward network (Input -> Hidden -> Output).
- **Initialization**: Weights initialized using normal distribution with scaling based on node count ($1/\sqrt{n}$).
- **Activation**: Sigmoid function via `scipy.special.expit`.
- **Learning**: Backpropagation algorithm with Stochastic Gradient Descent (SGD).
- **Custom Inference**: Includes a pipeline to process and predict custom handwritten digits from external PNG files (e.g., from MS Paint).
- **Compatibility**: Fully compatible with NumPy 2.x.

## Performance 
- **Dataset**: MNIST Handwritten Digits (CSV version).
- **Accuracy**: Achieves ~**96% accuracy** on the test set after 3-5 epochs (with 100 hidden nodes and 0.1 learning rate).

## Mathematical Implementation 
This implementation focuses on the core calculus of AI:
- Calculation of error gradients for each layer.
- Matrix transposition for backpropagating errors through the hidden layer.
- Weighted updates using the formula: $\Delta W = \alpha \cdot E \cdot O(1-O) \cdot I^T$.

## Getting Started 

### 1. Requirements
- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pillow`

```bash
pip install -r requirements.txt


![Скриншот Джарвиса](https://github.com/user-attachments/assets/441c8b8d-36db-4f20-b661-861e2ac29a46)


![Demo Image](https://github.com/user-attachments/assets/441c8b8d-36db-4f20-b661-861e2ac29a46)

<img width="1920" height="1039" alt="image" src="https://github.com/user-attachments/assets/441c8b8d-36db-4f20-b661-861e2ac29a46" />

