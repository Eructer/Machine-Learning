import numpy as np
from pathlib import Path
import time
import pandas as pd

from Layer import Layer
from Network import Network

def bench_net_size():
    print("Loading Data")
    df = pd.read_csv("data/file.csv")

    # Column order: price, quality, life, buy
    X = np.array(df.drop(columns=["clothing_class"]))  # features

    # 0 : t_shirt, 1 : long_sleeve, 2 : hoodie
    key = {0 : "t_shirt", 1 : "long_sleeve", 2 : "hoodie"}
    temp = []
    for i in df["clothing_class"].to_list():
        cloth_type = 0
        if i == "long_sleeve":
            cloth_type = 1
        elif i == "hoodie":
            cloth_type = 2
        else:
            cloth_type = 0
        temp.append(cloth_type)

    labels = np.array(temp) # target
    print("Splitting into train/test")
    # Train/test split
    def split(X, y, train_fraction=0.8):
        idx = np.random.permutation(len(X))
        cut = int(train_fraction * len(X))
        return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]

    classes = 3
    X_train, l_train, X_test, l_test = split(X, labels)

    l_train = l_train.astype(int)

    l_test = l_test.astype(int)

    y_train = np.eye(classes)[l_train]
    print("Build Network")
    # Build network : 25 -> 30 -> 30 -> 3
    net = Network()
    net.add_layer(Layer(25, 30, "lrelu"))
    net.add_layer(Layer(30,30, "lrelu"))
    net.add_layer(Layer(30, classes, "softmax"))

    def accuracy(net, X, labels):
        return np.mean(np.argmax(net.predict(X), axis=1) == labels)

    print(f"Accuracy prediction training: {accuracy(net, X_test, l_test):.1%}\n")

    start = time.time()
    print("Starting Training")
    # 3. Train
    losses = net.train(
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        early_stop=0.001,
        model_name="weather",
        save_path="models",
        save_model=False,
        print_loss_every=100,
    )

    end = time.time()

    print(f"\nTraining Time : {end - start}")

    print(f"Trained for {len(losses)} epochs, final loss {losses[-1]:.4f}")
    print(f"Train accuracy: {accuracy(net, X_train, np.argmax(l_train, axis=0)):.1%}")
    print(f"Test accuracy:  {accuracy(net, X_test, l_test):.1%}\n")

    # Test with new data
    point = [5,11,10,10,9,9,10,11,13,15,17,18,19,20,21,21,20,19,17,15,14,13,12,11,11]
    print(f"model: {point} -> class {net.predict_classes(point)[0]}, "
            f"probabilities {np.round(net.predict(point)[0], 3)}\nPick : {key.get(net.predict_classes(point)[0])}")