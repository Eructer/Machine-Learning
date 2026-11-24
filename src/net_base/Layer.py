import numpy


class Layer:
    """
    Fully connected layer used with Network to build an Artificial Neural Network.

    Attributes
    ----------
        input_size (int) : Number of inputs
        output_size (int) : Number of neurons
        activation_type (str) : "sigmoid", "linear", "lrelu", "softmax" or "none"
        weights, bias : Trainable parameters
        m_w, v_w, m_b, v_b, t : Adam optimizer state

    Methods
    -------
        forward (input_data) : Forward pass
        backward (grad) : Returns (input_error, dW, dB); does not modify parameters
    """

    ACTIVATIONS = ("sigmoid", "linear", "lrelu", "softmax", "none")

    def __init__(self, input_size: int, output_size: int, activation: str = "sigmoid") -> None:
        self.input_size = input_size
        self.output_size = output_size
        # Scaled random init keeps the signal size stable across layers
        self.weights = numpy.random.randn(input_size, output_size) * numpy.sqrt(1.0 / input_size)
        self.bias = numpy.zeros((1, output_size))
        # Adam optimizer state
        self.m_w = numpy.zeros_like(self.weights)
        self.v_w = numpy.zeros_like(self.weights)
        self.m_b = numpy.zeros_like(self.bias)
        self.v_b = numpy.zeros_like(self.bias)
        self.t = 0
        # Cached values from the last forward pass
        self.input = None
        self.output = None
        self._apply_activation(activation)

    def forward(self, input_data):
        """Forward pass: returns activation(input @ W + b)."""
        self.input = input_data
        z = numpy.dot(input_data, self.weights) + self.bias
        self.output = self.activation(z)
        return self.output

    def backward(self, grad):
        """
        Compute gradients for this layer.

        Parameters
        ----------
            grad : numpy.ndarray
                Gradient w.r.t. this layer's output. For a softmax layer trained
                with cross-entropy this is already (output - y_one_hot), i.e. the
                gradient w.r.t. the pre-activation, so no derivative is applied.

        Returns
        -------
            tuple : (input_error, dW, dB)
        """
        if self.activation_type == "softmax":
            delta = grad
        else:
            delta = grad * self.activation_derivative(self.output)
        m = delta.shape[0]
        dW = numpy.dot(self.input.T, delta) / m
        dB = numpy.sum(delta, axis=0, keepdims=True) / m
        # Uses the weights from the forward pass (before the optimizer step)
        input_error = numpy.dot(delta, self.weights.T)
        return input_error, dW, dB

    def _apply_activation(self, type: str) -> None:
        """Select the activation function and its derivative."""
        if not type:
            type = "linear"
        if type == "sigmoid":
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
            self.activation_type = "sigmoid"
        elif type in ("linear", "none"):
            self.activation = self._linear
            self.activation_derivative = self._linear_derivative
            self.activation_type = "linear" if type == "linear" else "none"
        elif type == "lrelu":
            self.activation = self._leaky_relu
            self.activation_derivative = self._leaky_relu_derivative
            self.activation_type = "lrelu"
        elif type == "softmax":
            self.activation = self._soft_max
            self.activation_derivative = None  # combined with cross-entropy loss
            self.activation_type = "softmax"
        else:
            raise ValueError(f"Unknown activation '{type}'. Choose from {self.ACTIVATIONS}")

    # Derivatives take the layer OUTPUT (a = f(z)), not z
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + numpy.exp(-numpy.clip(x, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def _linear(x):
        return x

    @staticmethod
    def _linear_derivative(a):
        return numpy.ones_like(a)

    @staticmethod
    def _leaky_relu(x, constant: float = 0.01):
        return numpy.where(x > 0, x, x * constant)

    @staticmethod
    def _leaky_relu_derivative(a, constant: float = 0.01):
        # Leaky ReLU preserves sign, so a < 0 exactly when z < 0
        return numpy.where(a > 0, 1.0, constant)

    @staticmethod
    def _soft_max(x):
        exp_logits = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
        return exp_logits / numpy.sum(exp_logits, axis=1, keepdims=True)
