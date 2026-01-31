import argparse
import numpy as np
from neural_network import NeuralNetwork


def parse_mnist_record(line: str, output_nodes: int):
    parts = line.strip().split(",")
    label = int(parts[0])

    pixels = np.asarray(parts[1:], dtype=np.float32)
    inputs = (pixels / 255.0 * 0.99) + 0.01

    targets = np.full(output_nodes, 0.01, dtype=np.float32)
    targets[label] = 0.99

    return inputs, targets, label


def evaluate(net: NeuralNetwork, test_path: str, output_nodes: int) -> float:
    correct = 0
    total = 0

    with open(test_path, "r") as f:
        for line in f:
            inputs, _, label = parse_mnist_record(line, output_nodes)
            outputs = net.query(inputs)
            prediction = int(np.argmax(outputs))
            correct += int(prediction == label)
            total += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="mnist_train.csv")
    parser.add_argument("--test", default="mnist_test.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=100)
    args = parser.parse_args()

    input_nodes = 784
    output_nodes = 10

    net = NeuralNetwork(input_nodes, args.hidden, output_nodes, args.lr)

    for epoch in range(args.epochs):
        with open(args.train, "r") as f:
            for line in f:
                inputs, targets, _ = parse_mnist_record(line, output_nodes)
                net.train(inputs, targets)

        acc = evaluate(net, args.test, output_nodes)
        print(f"Epoch {epoch + 1}/{args.epochs} | accuracy={acc:.4f}")


if __name__ == "__main__":
    main()