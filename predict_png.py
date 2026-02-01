import argparse
import numpy as np
from PIL import Image
from neural_network import NeuralNetwork


def load_png_as_inputs(path: str, debug_path: str = "debug_28x28.png"):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

   
    arr = 255 - arr

    thresh = 50
    arr = (arr > thresh).astype(np.uint8) * 255

    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty image after threshold (no digit pixels found).")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    digit = arr[y1:y2+1, x1:x2+1]

    target = 20
    h, w = digit.shape
    scale = target / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit_img = Image.fromarray(digit)
    digit_img = digit_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    digit_small = np.asarray(digit_img, dtype=np.uint8)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_small

    Image.fromarray(canvas).save(debug_path)

    inputs = (canvas.flatten().astype(np.float32) / 255.0 * 0.99) + 0.01
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="mnist_weights.npz")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    net = NeuralNetwork.load(args.weights)
    inputs = load_png_as_inputs(args.image)

    outputs = net.query(inputs)
    prediction = int(np.argmax(outputs))
    confidence = float(outputs[prediction, 0])

    print("Prediction:", prediction)
    print("Confidence (not real confidence):", round(confidence, 4))
    print("Outputs:", outputs.T)

if __name__ == "__main__":
    main()