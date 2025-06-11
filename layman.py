# Import dependencies
import numpy as np
import pandas as pd
from typing import Literal
from random import uniform
from dataclasses import dataclass

"""
Node or neuron

Holds following values:
    1. val - value
    2. conn_weight - weight per connection (type list)
    3. epoch_data - saves changes after each epoch
    4. pre_activation
"""
@dataclass
class Node:
    val: float
    conn_weight: list
    epoch_data: list = None
    pre_activation: float = 0

"""
Network

Represent complete neural structure and follows the give hierarchy
--Network
    |
    |____Input Layer
            |
            |__Node
            |__Node
            |__ ......
    _____Hidden Layer
        |
        |__Node
        |__Node
        |__ ......

    and further
"""
class Network:
    def __init__(self):
        super().__init__()
        self.layers = []
        self.bias = 0

    """
    create_layer

    creates a layer which holds neurons
    number of neuron depends upon the type of data and has to be provided by the user
    nodes are automatically created during layer creation and random weights are assigned
    """
    def create_layer(self, nodes_count: int, layer_name: Literal["input", "hidden", "output"] = "hidden"):
        nodes = []
        for i in range(nodes_count):
            node = Node(uniform(0.5, 1), [])
            nodes.append(node)
        layer = {"name": layer_name, "nodes": nodes}
        self.layers.append(layer)

    """
    get_layer

    returns the layer from self.layer
    the returned layer contains all the assosciated nodes
    """
    def get_layer(self, layer_name: Literal["input", "hidden", "output"] = "hidden"):
        return [layer for layer in self.layers if layer['name'] == layer_name]
    
    
    """
    forward_propogate

    performs tradition forward propogation
    """
    def forward_propagate(self):
        input_layer = self.get_layer("input")[0]
        hidden_layers = self.get_layer("hidden")
        output_layer = self.get_layer("output")[0]

        hidden_layer = hidden_layers[0] if hidden_layers else None

        for h_index, h_node in enumerate(hidden_layer['nodes']):
            total_input = 0
            for i_index, i_node in enumerate(input_layer['nodes']):
                weight = i_node.conn_weight[h_index]
                total_input += i_node.val * weight
            h_node.pre_activation = total_input + self.bias
            h_node.val = self.relu(h_node.pre_activation)

        for o_index, o_node in enumerate(output_layer['nodes']):
            total_output = 0
            for h_index, h_node in enumerate(hidden_layer['nodes']):
                weight = h_node.conn_weight[o_index]
                total_output += h_node.val * weight
            o_node.pre_activation = total_output + self.bias
            o_node.val = o_node.pre_activation


    """
    adjust_network

    assigns weights to following connected nodes in the next layer
    """
    def adjust_network(self):
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
    
            num_nodes_current = len(current_layer['nodes'])
            num_nodes_next = len(next_layer['nodes'])
    
            for node in current_layer['nodes']:
                node.conn_weight = [uniform(-0.5, 0.5) for _ in range(num_nodes_next)]

    def relu(self, x):
        return max(0, x)
    

    """
    predict

    Predicts output
    """
    def predict(self, input_vals):
        for i, val in enumerate(input_vals):
            self.layers[0]['nodes'][i].val = val
        self.forward_propagate()
        output_layer = self.get_layer("output")[0]
        outputs = [node.val for node in output_layer['nodes']]
        return np.argmax(outputs), outputs
    

    """
    calc_mse

    calculates mean squared error
    """
    def calc_mse(self, actual, predicted):
        mse = 0.5 * sum((a - p) ** 2 for a, p in zip(actual, predicted))
        return mse


    """used in calculating derivatives for backward propogation"""
    def calc_derivatives(self, loss, actual, predicted):
        d_loss = 2 * (predicted - actual)
        
    """
    backward_propogate

    performs traditional backward propogation
    """
    def backward_propagate(self, actual_outputs):
        # Get layers
        input_layer = self.get_layer("input")[0]
        hidden_layer = self.get_layer("hidden")[0]
        output_layer = self.get_layer("output")[0]
    
        # Calculate delta for output nodes (since output is linear, derivative of activation = 1)
        for i, o_node in enumerate(output_layer['nodes']):
            error = o_node.val - actual_outputs[i]
            o_node.delta = error  # dL/dz = (predicted - actual) for MSE
    
        # Calculate delta for hidden nodes
        for i, h_node in enumerate(hidden_layer['nodes']):
            downstream_gradient = 0
            for j, o_node in enumerate(output_layer['nodes']):
                downstream_gradient += o_node.delta * h_node.conn_weight[j]
            # Derivative of ReLU
            relu_derivative = 1 if h_node.pre_activation > 0 else 0
            h_node.delta = downstream_gradient * relu_derivative
    
        # Update weights from hidden to output layer
        learning_rate = 0.01
        for h_node in hidden_layer['nodes']:
            for j, o_node in enumerate(output_layer['nodes']):
                grad = o_node.delta * h_node.val  # dL/dw
                h_node.conn_weight[j] -= learning_rate * grad
    
        # Update weights from input to hidden layer
        for i_node in input_layer['nodes']:
            for j, h_node in enumerate(hidden_layer['nodes']):
                grad = h_node.delta * i_node.val
                i_node.conn_weight[j] -= learning_rate * grad
    
        # Optionally update bias terms if you want
        self.bias -= learning_rate * sum(o_node.delta for o_node in output_layer['nodes'])

    def train(self, X_train, y_train, epochs=10, verbose=True):
        loss_vals = []
        for epoch in range(epochs):
            export_to_csv(self, f"epoch_{epoch}.csv")
            total_loss = 0
            for i in range(len(X_train)):
                x = X_train[i]
                y_true = y_train[i]  # assumed to be one-hot encoded
    
                pred_class, output = self.predict(x)
                loss = self.calc_mse(y_true, output)
                total_loss += loss
                loss_vals.append(loss)
    
                self.backward_propagate(y_true)
    
            if verbose:
                avg_loss = total_loss / len(X_train)
                print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        return loss_vals


"""
export_to_csv

collect data of each neuron per epoch and log it.
"""
def export_to_csv(network: Network, filename: str):
    # only accounts for neurons and it's values i.e weight for each connection and value
    data = []
    input_layer = network.get_layer("input")[0]
    hidden_layer = network.get_layer("hidden")[0]
    output_layer = network.get_layer("output")[0]

    node_ids = []
    node_vals = []
    node_weights = []

    for node_idx, node in enumerate(input_layer['nodes']):
        float_weights = [float(weight) for weight in node.conn_weight]
        
        node_ids.append(f"i{node_idx}")
        node_vals.append(node.val)
        node_weights.append(float_weights)

    data = {
        "node_id": node_ids,
        "node_val": node_vals,
        "node_weights": node_weights
    }

    data_df = pd.DataFrame.from_dict(data)
    data_df.to_csv(filename, index=False)