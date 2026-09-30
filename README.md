# FSANN

**FSANN -> From Scratch Artificial Neural Network**

This repo contains my attempt at making a Artificial Neural Network from scratch

Meaning no ML/DL libraries used, I did use numpy for its matrix functions

# Architecture

I have two classes `Network` and `Layer`, each in their respective file

You use multiple `Layer`s to make a `Network` (you can't define a `Network` without any `Layer`s)

I also wanted this to be able to handle both regression and classification tasks

## Layer

The `Layer` class represents the layers that make up the `Network`. I did not choose to go with individual neurons or a neuron class, because they would all eventually end up in a `Layer` representation.

### Attributes

- `input_size` : How many inputs are going to be coming into the `Layer`
- `output_size` : How many outputs are coming out the `Layer`
- `activation_type` : What kind of activation function to use for the layer

### Methods

- `forward` : Perform a forward pass through the `Layer`
- `backward` : Perform a backward pass through the `Layer`, for gradient decent

### Usage

Constructing a `Layer`

```python
from Layer import Layer

Layer(input_size = 3, output_size = 5, activation_type = "lrelu")
```

## Network

The `Network` class represents multiple `Layers` that make up the `Network`

From a `Network` object you can build a complete `Network` with `Layer`s, train the network on data, and you can save the trained model

### Attributes

- `model_path` : Model save path/destination

### Methods

- `add_layer` : Add a layer to the `Network`
- `forward` : Perform a forward pass through the whole `Network`
- `predict` : Perform a prediction (regression)
- `predict_classes` : Perform a class/label prediction (classification)
- `backward` : Perform a backward pass through the whole `Network`
- `train` : Train the model/`Network`
- `save` : Save the model/`Network`
- `load_model` : Load a model/`Network` from a save file

### Usage

Constructing a full/complete `Network`

```python
from Network import Network
from Layer import Layer

net = Network()
net.add_layer(Layer(3,5,"lrelu"))
net.add_layer(Layer(5,5,"lrelu"))
net.add_layer(Layer(5,2,"softmax"))

```

---

Regression predicting with `Network`, _assuming you already trained_

```python
from Network import Network
from Layer import Layer

# Construct
net = Network()
net.add_layer(Layer(3,5,"lrelu"))
net.add_layer(Layer(5,5,"lrelu"))
net.add_layer(Layer(5,2,"softmax"))

# Train step inbetween

# Get new data 
new_data = [..., ...]
# Predict
prediction = net.predict(new_data)

```


# Model/Network Saving

To use you just need to set `save_model = True` when you call the training function

```python
from Network import Network
from Layer import Layer

net = Network()
net.add_layer(Layer(3,5,"lrelu"))
net.add_layer(Layer(5,5,"lrelu"))
net.add_layer(Layer(5,2,"softmax"))

losses = net.train(.., .., .., .., .., .., .., save_model=True, ..)
```

Set `save_model = False` if you don't want to save

---

To save the model/`Network`, the easiest thing I could think of is using `json`

When saving the model/`Network` it iterates through the trained `Networks` `Layer`s pulling out the data

- Weights
- Bias
- Activation type
- input size
- output size

It save this data for a `Layer` as a `dict`, then puts the `dict` in a `Layer` list

After iterating through all `Layer`s, the `Layers` list is put in another `dict`, which becomse the final `dict`

After creating the final `dict` its written to a file with `json.dump`

```python
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
```

---

For small `Networks` this is a totally viable solution, but for larger `Networks` the file size will/could be a problem (and at that point just use a actual ML/DL library)

# Loading a saved model
To load a model/`Network` from a save file

```python
loaded = Network()

loaded.load_model("path/to/save_file.json")
```

---

For loading a saved model/`Network` the process is just the opposite when saving one, the program just iterates through the `Layer`s

As it loads the `Layer` data it constructs the `Layer` and adds it to the network (similar to how a `Network` gets constructed)

```python
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
```

# Examples

## Classification

In this example I created fake data about chapsticks (I go through a lot of chapsticks) and would like to see that given some features about a chapstick brand, should I buy one

This is a classification problem

Fake data

```python
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
```

The full example is [chap.py](src/chap.py)


## Regression

In this one I tested out regression with a equation that the `Network` needs to find the variables for (aka weights and bias)

Target equation

$y = 3 * 1.0 - 2 * -1.0 + 0.5 ** 2$

The full example is [regression.py](src/regression.py)
