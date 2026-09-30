import numpy as np

from Layer import Layer
from Network import Network

def split(X, y, train_fraction=0.8):
    idx = np.random.permutation(len(X))
    cut = int(train_fraction * len(X))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 50)
    print("REGRESSION: 3 features -> 1 continuous target")
    print("=" * 50)
    n = 500
    X = np.random.uniform(-2, 2, size=(n, 3))
    # Nonlinear ground truth with a little noise
    y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(n) * 0.1

    X_train, y_train, X_test, y_test = split(X, y)

    # Standardize features AND target (helps training a lot); reuse train statistics
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(), y_train.std()
    Xs_train, Xs_test = (X_train - x_mean) / x_std, (X_test - x_mean) / x_std
    ys_train = (y_train - y_mean) / y_std

    net = Network()
    net.add_layer(Layer(3, 32, "lrelu"))
    net.add_layer(Layer(32, 16, "lrelu"))
    net.add_layer(Layer(16, 1, "linear"))  # linear output -> MSE (chosen automatically)

    net.train(Xs_train, ys_train, epochs=500, learning_rate=0.005,
              early_stop=0.005, print_loss_every=100)

    # Predict, then convert back to the original target scale
    pred = net.predict(Xs_test).ravel() * y_std + y_mean
    mse = np.mean((pred - y_test) ** 2)
    r2 = 1 - np.sum((pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"Test MSE: {mse:.4f}   R^2: {r2:.3f}")

    sample = np.array([[1.0, -1.0, 0.5]])
    value = net.predict((sample - x_mean) / x_std)[0, 0] * y_std + y_mean
    truth = 3 * 1.0 - 2 * -1.0 + 0.5 ** 2
    print(f"Sample {sample[0].tolist()} -> predicted {value:.3f}, true {truth:.3f}")