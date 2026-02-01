# Neural Network from Scratch (MNIST)

A small feed-forward neural network implemented from scratch (NumPy + SciPy), trained on the MNIST handwritten digits dataset (CSV version).

No TensorFlow / PyTorch — just the math.

## Features
- 3-layer neural network: input → hidden → output
- Sigmoid activation
- Backpropagation + gradient descent
- MNIST training + evaluation with accuracy per epoch
- Works with NumPy 2.x

## Requirements
- Python 3.10+ recommended

Install dependencies:
```bash
pip install -r requirements.txt


## How to run:
To train the model: python train_mnist.py
To predict your digit: python predict_png.py --image my.png
