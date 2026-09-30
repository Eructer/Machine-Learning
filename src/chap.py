import numpy as np
from pathlib import Path
from Layer import Layer
from Network import Network

# Chapstick
# price | quality (1-5) | life (days) | Buy (1 or 0)
#   1.5  |      5        |     20      |     1
#   1.5  |      1        |     20      |     0
#   2.5  |      4        |     30      |     1
#   1.0  |      3        |     15      |     1
#   3.5  |      2        |     10      |     0
#   2.0  |      4        |     25      |     1
#   1.5  |      2        |      5      |     0
#   4.0  |      5        |     30      |     1
#   2.5  |      3        |     15      |     0
#   1.0  |      5        |     30      |     1
#   3.0  |      1        |      1      |     0
#   2.0  |      2        |     10      |     0
#   1.5  |      4        |     25      |     1
#   3.5  |      5        |     30      |     1
#   2.5  |      1        |      5      |     0
#   1.0  |      2        |      1      |     0
#   4.5  |      3        |     20      |     0
#   2.0  |      5        |     30      |     1
#   3.0  |      4        |     15      |     1
#   1.5  |      3        |     10      |     0
#   5.0  |      1        |      1      |     0
#   2.5  |      5        |     30      |     1
#   1.0  |      4        |     20      |     1
#   3.5  |      2        |      5      |     0
#   2.0  |      3        |     30      |     1   

if __name__ == "__main__":
    model_path = Path("models/chapstick.json")

    data = np.array([
        [1.5, 5, 20, 1],
        [1.5, 1, 20, 0],
        [2.5, 4, 30, 1],
        [1.0, 3, 15, 1],
        [3.5, 2, 10, 0],
        [2.0, 4, 25, 1],
        [1.5, 2,  5, 0],
        [4.0, 5, 30, 1],
        [2.5, 3, 15, 0],
        [1.0, 5, 30, 1],
        [3.0, 1,  1, 0],
        [2.0, 2, 10, 0],
        [1.5, 4, 25, 1],
        [3.5, 5, 30, 1],
        [2.5, 1,  5, 0],
        [1.0, 2,  1, 0],
        [4.5, 3, 20, 0],
        [2.0, 5, 30, 1],
        [3.0, 4, 15, 1],
        [1.5, 3, 10, 0],
        [5.0, 1,  1, 0],
        [2.5, 5, 30, 1],
        [1.0, 4, 20, 1],
        [3.5, 2,  5, 0],
        [2.0, 3, 30, 1],
    ])

    # Column order: price, quality, life, buy
    X = data[:, :3]  # features
    labels = data[:, 3] # target

    # Train/test split
    def split(X, y, train_fraction=0.8):
        idx = np.random.permutation(len(X))
        cut = int(train_fraction * len(X))
        return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]

    classes = 2
    X_train, l_train, X_test, l_test = split(X, labels)

    l_train = l_train.astype(int)

    l_test = l_test.astype(int)

    y_train = np.eye(classes)[l_train]

    # Build network : 3 -> 5 -> 5 -> 1
    net = Network()
    net.add_layer(Layer(3, 5, "lrelu"))
    net.add_layer(Layer(5,5, "lrelu"))
    net.add_layer(Layer(5, classes, "softmax"))

    def accuracy(net, X, labels):
        return np.mean(np.argmax(net.predict(X), axis=1) == labels)

    print(f"Accuracy before training: {accuracy(net, X_test, l_test):.1%}\n")

    # 3. Train (and save)
    losses = net.train(
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        early_stop=0.05,
        model_name="chapstick",
        save_path="models",
        save_model=False,
        print_loss_every=100,
    )
    print(f"\nTrained for {len(losses)} epochs, final loss {losses[-1]:.4f}")
    print(f"Train accuracy: {accuracy(net, X_train, np.argmax(l_train, axis=0)):.1%}")
    print(f"Test accuracy:  {accuracy(net, X_test, l_test):.1%}\n")

    # Test with new data
    point = [0.5, 2, 40]
    print(f"model: {point} -> class {net.predict_classes(point)[0]}, "
            f"probabilities {np.round(net.predict(point)[0], 3)}\n")
    print(f"Buy, confidence : {np.round(loaded.predict(point)[0], 3)[1]}" if loaded.predict_classes(point)[0] else f"Don't Buy : confidence {np.round(loaded.predict(point)[0], 3)[0]}")

