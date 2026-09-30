import json
import os
import time

import numpy

from Layer import Layer


class Network:
    """
    Artificial Neural Network (mini-batch training, Adam optimizer).

    Classification: last layer activation="softmax", one-hot y, cross-entropy loss.
    Regression:     last layer activation="linear",  numeric y, mean squared error loss.

    Methods
    -------
        add_layer (layer) : Add a layer
        forward (X) : Forward pass
        predict (input_data) : Class probabilities (classification) or values (regression)
        predict_classes (input_data) : Most likely class index per sample
        backward (grad, learning_rate) : Backpropagate and update parameters
        train (X, y, epochs, learning_rate, loss, ...) : Train for either task
        save (file_name, destination) : Save a model to .json
        load_model (path) : Load a model from .json
    """

    def __init__(self, model_path: str = "") -> None:
        self.layers = []
        self.file_path = model_path

    def add_layer(self, layer: Layer) -> None:
        """Append a layer to the network."""
        self.layers.append(layer)

    def forward(self, X):
        """Forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, input_data):
        """
        Make a prediction. Accepts a single sample (1D) or a batch (2D).

        Returns
        -------
            numpy.ndarray : Shape (n_samples, n_outputs). Class probabilities or
            predicted values, depending on the output layer.
        """
        x = numpy.atleast_2d(numpy.array(input_data, dtype=float))
        return self.forward(x)

    def predict_classes(self, input_data):
        """Return the most likely class index for each sample (classification)."""
        return numpy.argmax(self.predict(input_data), axis=1)

    def backward(self, grad, learning_rate: float, clip_value: float = 1.0) -> None:
        """
        Backpropagate the loss gradient and update every layer with Adam.

        Parameters
        ----------
            grad : numpy.ndarray
                Gradient of the loss w.r.t. the output layer's pre-activation
                (output - y_one_hot for softmax + cross-entropy)
            learning_rate : float
            clip_value : float
        """
        for layer in reversed(self.layers):
            # Gradients are computed with the pre-update weights
            grad, dW, dB = layer.backward(grad)
            dW = numpy.clip(dW, -clip_value, clip_value)
            dB = numpy.clip(dB, -clip_value, clip_value)
            self._adam_update(layer, dW, dB, learning_rate)

    def train(
        self,
        X,
        y,
        epochs: int,
        learning_rate: float,
        early_stop: float = 0.01,
        loss: str = "auto",
        model_name: str = "",
        save_path: str = "",
        save_model: bool = False,
        print_loss_every: int = 100,
    ) -> list:
        """
        Train the network for classification or regression.

        Parameters
        ----------
            X : numpy.ndarray, shape (n_samples, n_features)
            y : numpy.ndarray
                Classification: one-hot labels, shape (n_samples, n_classes)
                Regression: targets, shape (n_samples,) or (n_samples, n_outputs)
            epochs : int
            learning_rate : float
            early_stop : float
                Stop when the average epoch loss drops to this value
            loss : str
                "cross_entropy" (last layer must be softmax),
                "mse" (last layer must NOT be softmax, usually "linear"),
                or "auto" (softmax last layer -> cross_entropy, otherwise mse)
            model_name, save_path, save_model : saving options
            print_loss_every : int
                Print every N epochs (0 disables printing)

        Returns
        -------
            list : Average loss for each epoch
        """
        if not self.layers:
            raise ValueError("Network has no layers")

        is_softmax = self.layers[-1].activation_type == "softmax"
        if loss == "auto":
            loss = "cross_entropy" if is_softmax else "mse"
        if loss not in ("cross_entropy", "mse"):
            raise ValueError("loss must be 'auto', 'cross_entropy' or 'mse'")
        if loss == "cross_entropy" and not is_softmax:
            raise ValueError("cross_entropy needs the last layer to use activation='softmax'")
        if loss == "mse" and is_softmax:
            raise ValueError("mse needs a non-softmax last layer (e.g. activation='linear')")

        # A 1-D regression target becomes a column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if y.shape[1] != self.layers[-1].output_size:
            raise ValueError(
                f"y has {y.shape[1]} column(s) but the last layer has "
                f"{self.layers[-1].output_size} output(s)"
            )

        losses = []
        num_samples = X.shape[0]
        indices = numpy.arange(num_samples)
        batch_size = self._get_batch_size(num_samples)

        for epoch in range(epochs):
            start_time = time.time()
            # Shuffle indices so (x, y) pairs stay together
            numpy.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = self.forward(X_batch)
                m = y_batch.shape[0]

                if loss == "cross_entropy":
                    # Clipped to avoid log(0)
                    y_indices = numpy.argmax(y_batch, axis=1)
                    probs = numpy.clip(output[numpy.arange(m), y_indices], 1e-12, 1.0)
                    batch_loss = -numpy.sum(numpy.log(probs)) / m
                else:
                    # 0.5 * squared error, summed over outputs, averaged over samples
                    batch_loss = 0.5 * numpy.sum((output - y_batch) ** 2) / m

                epoch_loss += batch_loss * m

                # Both losses give a gradient of (output - y):
                #   cross_entropy: w.r.t. the softmax pre-activation
                #   mse: w.r.t. the output (Layer.backward applies the activation derivative)
                self.backward(output - y_batch, learning_rate)

            epoch_loss /= num_samples
            losses.append(epoch_loss)

            if print_loss_every != 0 and epoch % print_loss_every == 0:
                print(f"Epoch {epoch}, {loss} loss: {epoch_loss:.8f} ({time.time() - start_time:.3f}s)")

            if epoch_loss <= early_stop:
                print(f"Early stop at epoch {epoch}, loss: {epoch_loss:.8f}")
                break

        if save_model:
            self._save_model(model_name, save_path)
            print(f"Saved {model_name} successfully")

        return losses

    def save(self, file_name: str, destination: str = "") -> None:
        """Save the model to <destination>/<file_name>.json."""
        self._save_model(file_name, destination)

    def _save_model(self, file_name: str, destination: str) -> None:
        """Save the model as JSON. Raises on failure."""
        layers_data = []
        for i, layer in enumerate(self.layers):
            layers_data.append(
                {
                    f"layer {i}": {
                        "inputs": layer.input_size,
                        "outputs": layer.output_size,
                        "activation": layer.activation_type,
                        "weights": layer.weights.tolist(),
                        "bias": layer.bias.tolist(),
                    }
                }
            )
        if destination:
            os.makedirs(destination, exist_ok=True)
        path = os.path.join(destination, f"{file_name}.json")
        with open(path, "w") as file:
            json.dump({"model": layers_data}, file, indent=4)
        self.file_path = path

    def load_model(self, path: str = "") -> bool:
        """
        Load a saved model (.json). Uses self.file_path if no path is given.

        Returns
        -------
            bool : True on success (raises on failure)
        """
        data = self._load_model(path or self.file_path)
        if not data or "model" not in data:
            raise ValueError("Invalid or missing model data")

        self.layers.clear()
        for layer_dict in data["model"]:
            layer_info = next(iter(layer_dict.values()))
            layer = Layer(
                layer_info["inputs"],
                layer_info["outputs"],
                layer_info.get("activation", "sigmoid"),
            )
            layer.weights = numpy.array(layer_info["weights"])
            layer.bias = numpy.array(layer_info["bias"])
            # Optimizer state must match the loaded parameter shapes
            layer.m_w = numpy.zeros_like(layer.weights)
            layer.v_w = numpy.zeros_like(layer.weights)
            layer.m_b = numpy.zeros_like(layer.bias)
            layer.v_b = numpy.zeros_like(layer.bias)
            self.add_layer(layer)
        return True

    @staticmethod
    def _load_model(file_path: str) -> dict:
        """Read a model .json file."""
        if file_path == "":
            raise ValueError("Empty File Path")
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}") from e

    @staticmethod
    def _get_batch_size(num_samples, fraction=0.1, min_size=4, max_size=128) -> int:
        """Batch size: a fraction of the data, clamped, rounded to a power of 2."""
        batch_size = int(num_samples * fraction)
        batch_size = max(min_size, min(batch_size, max_size))
        return 2 ** int(round(numpy.log2(batch_size)))

    @staticmethod
    def _adam_update(
        layer,
        dW,
        dB,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ) -> None:
        """Adam optimizer step: momentum + adaptive per-parameter learning rates."""
        layer.t += 1
        layer.m_w = beta1 * layer.m_w + (1 - beta1) * dW
        layer.m_b = beta1 * layer.m_b + (1 - beta1) * dB
        layer.v_w = beta2 * layer.v_w + (1 - beta2) * (dW ** 2)
        layer.v_b = beta2 * layer.v_b + (1 - beta2) * (dB ** 2)
        # Bias correction
        m_w_hat = layer.m_w / (1 - beta1 ** layer.t)
        m_b_hat = layer.m_b / (1 - beta1 ** layer.t)
        v_w_hat = layer.v_w / (1 - beta2 ** layer.t)
        v_b_hat = layer.v_b / (1 - beta2 ** layer.t)
        layer.weights -= learning_rate * m_w_hat / (numpy.sqrt(v_w_hat) + epsilon)
        layer.bias -= learning_rate * m_b_hat / (numpy.sqrt(v_b_hat) + epsilon)
