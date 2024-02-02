"""
**Author:** Matthieu DE MARI (matthieu_demari@sutd.edu.sg)

**Version:** 1.1 (16/06/2023)

**Requirements:**
- Python 3 (tested on v3.11.4)
- Matplotlib (tested on v3.7.1)
- Numpy (tested on v1.24.3)

**Description:** This file was used in the 50.039 Deep Learning course at the Singapore University of Technology and Design. 
It serves as support for the W2S3 Notebook 8 Practice Activities.
Feel free to epxlore the functions, but modify them at your own risk!
"""

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Sklearn
from sklearn.metrics import accuracy_score
# Removing unecessary warnings (optional, just makes notebook outputs more readable)
import warnings
warnings.filterwarnings("ignore")

"""
Dataset Generator Functions
"""

def val(min_val, max_val):
    return round(np.random.uniform(min_val, max_val), 2)
    
def mysterious_equation(val1, val2):
    return 2*val1**2 + 3*val2**2 + 0.75*val1 + 0.25*val2
    
def class_for_val(val1, val2):
    return int(mysterious_equation(val1, val2) >= 1)
    
def create_dataset(n_points, min_val, max_val):
    val1_list = np.array([val(min_val, max_val) for _ in range(n_points)])
    val2_list = np.array([val(min_val, max_val) for _ in range(n_points)])
    inputs = np.array([[v1, v2] for v1, v2 in zip(val1_list, val2_list)])
    outputs = np.array([class_for_val(v1, v2) for v1, v2 in zip(val1_list, val2_list)]).reshape(n_points, 1)
    return val1_list, val2_list, inputs, outputs

def show_dataset(val1_list, val2_list, outputs):
    # Initialize plot
    fig = plt.figure(figsize = (10, 7))
    
    # Scatter plot
    markers = {0: "x", 1: "o"}
    colors = {0: "r", 1: "g"}
    indexes_0 = np.where(outputs == 0)[0]
    v1_0 = val1_list[indexes_0]
    v2_0 = val2_list[indexes_0]
    indexes_1 = np.where(outputs == 1)[0]
    v1_1 = val1_list[indexes_1]
    v2_1 = val2_list[indexes_1]
    plt.scatter(v1_0, v2_0, c = colors[0], marker = markers[0])
    plt.scatter(v1_1, v2_1, c = colors[1], marker = markers[1])
    
    # Show
    plt.show()

def show_dataset_and_predictions(inputs, val1_list, val2_list, outputs, model):
    # Initialize plot
    fig = plt.figure(figsize = (10, 7))
    
    # --- Scatter dataset plot
    markers = {0: "s", 1: "o"}
    colors = {0: "r", 1: "g"}
    indexes_0 = np.where(outputs == 0)[0]
    v1_0 = val1_list[indexes_0]
    v2_0 = val2_list[indexes_0]
    indexes_1 = np.where(outputs == 1)[0]
    v1_1 = val1_list[indexes_1]
    v2_1 = val2_list[indexes_1]
    plt.scatter(v1_0, v2_0, c = colors[0], marker = markers[0])
    plt.scatter(v1_1, v2_1, c = colors[1], marker = markers[1])
    
    # --- Make predictions 
    outputs_pred = np.round(model.forward(inputs))
    
    # --- Scatter prediction plot
    markers_pred = {0: "P", 1: "*"}
    colors_pred = {0: "b", 1: "k"}
    indexes_0_pred = np.where(outputs_pred == 0)[0]
    v1_0_pred = val1_list[indexes_0_pred]
    v2_0_pred = val2_list[indexes_0_pred]
    indexes_1_pred = np.where(outputs_pred == 1)[0]
    v1_1_pred = val1_list[indexes_1_pred]
    v2_1_pred = val2_list[indexes_1_pred]
    plt.scatter(v1_0_pred, v2_0_pred, c = colors_pred[0], marker = markers_pred[0])
    plt.scatter(v1_1_pred, v2_1_pred, c = colors_pred[1], marker = markers_pred[1])
    
    # --- Show legend
    legend_elements = [Line2D([0], [0], marker = 's', color = 'w', label = 'True 0', \
                              markerfacecolor = 'r', markersize = 10),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'True 1', \
                              markerfacecolor = 'g', markersize = 10),
                       Line2D([0], [0], marker = 'P', color = 'w', label = 'Predicted 0', \
                              markerfacecolor = 'b', markersize = 10),
                       Line2D([0], [0], marker = '*', color = 'w', label = 'Predicted 1', \
                              markerfacecolor = 'k', markersize = 10)]
    plt.legend(handles = legend_elements, loc = 'best')
    acc = accuracy_score(outputs_pred, outputs)
    plt.title("Model Predictions on Dataset - Accuracy Score = {}".format(acc))
    plt.show()
    

"""
Single-layer Neural Network class
"""

class ShallowNeuralNet_WithAct_OneLayer():
    
    def __init__(self, n_x, n_y):
        # Network dimensions
        self.n_x = n_x
        self.n_y = n_y
        # Initialize parameters
        self.init_parameters_normal()
        # Loss, initialized as infinity before first calculation is made
        self.loss = float("Inf")
         
    def init_parameters_normal(self):
        # Weights and biases matrices (randomly initialized)
        self.W = np.random.randn(self.n_x, self.n_y)*0.1
        self.b = np.random.randn(1, self.n_y)*0.1

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))
    
    def forward(self, inputs):
        # Wx + b operation for the second layer
        Z = np.matmul(inputs, self.W)
        Z_b = Z + self.b
        y_pred = self.sigmoid(Z_b)
        return y_pred
    
    def CE_loss(self, inputs, outputs):
        # CE loss function as before
        outputs_re = outputs.reshape(-1, 1)
        pred = self.forward(inputs)
        eps = 1e-10
        losses = outputs*np.log(pred + eps) + (1 - outputs)*np.log(1 - pred + eps)
        self.loss = -np.sum(losses)/outputs.shape[0]
        return self.loss
    
    def backward(self, inputs, outputs, alpha = 1e-5):
        # Get the number of samples in dataset
        m = inputs.shape[0]
        
        # Forward propagate
        Z = np.matmul(inputs, self.W)
        Z_b = Z + self.b
        A = self.sigmoid(Z_b)
    
        # Compute error term
        dL_dA = -outputs/A + (1 - outputs)/(1 - A)
        dL_dZ = dL_dA*A*(1 - A)
        
        # Gradient descent update rules
        self.W -= (1/m)*alpha*np.dot(A.T, dL_dZ)
        self.b -= (1/m)*alpha*np.sum(dL_dZ, axis = 0, keepdims = True)
        
        # Update loss
        self.CE_loss(inputs, outputs)
    
    def train(self, inputs, outputs, N_max = 1000, alpha = 1e-5, delta = 1e-5, display = True):
        # List of losses, starts with the current loss
        self.losses_list = [self.loss]
        # Repeat iterations
        for iteration_number in range(1, N_max + 1):
            # Backpropagate
            self.backward(inputs, outputs, alpha)
            new_loss = self.loss
            # Update losses list
            self.losses_list.append(new_loss)
            # Display
            if(display and iteration_number % (N_max*0.05) == 1):
                print("Iteration {} - Loss = {}".format(iteration_number, new_loss))
            # Check for delta value and early stop criterion
            difference = abs(self.losses_list[-1] - self.losses_list[-2])
            if(difference < delta):
                if(display):
                    print("Stopping early - loss evolution was less than delta on iteration {}.".format(iteration_number))
                break
        else:
            # Else on for loop will execute if break did not trigger
            if(display):
                print("Stopping - Maximal number of iterations reached.")
    
    def show_losses_over_training(self):
        # Initialize matplotlib
        fig, axs = plt.subplots(1, 2, figsize = (15, 5))
        axs[0].plot(list(range(len(self.losses_list))), self.losses_list)
        axs[0].set_xlabel("Iteration number")
        axs[0].set_ylabel("Loss")
        axs[1].plot(list(range(len(self.losses_list))), self.losses_list)
        axs[1].set_xlabel("Iteration number")
        axs[1].set_ylabel("Loss (in logarithmic scale)")
        axs[1].set_yscale("log")
        # Display
        plt.show()

"""
Two-layers Neural Network class
"""

class ShallowNeuralNet_WithAct_TwoLayers():
    
    def __init__(self, n_x, n_h, n_y):
        # Network dimensions
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        # Initialize parameters
        self.init_parameters_normal()
        # Loss, initialized as infinity before first calculation is made
        self.loss = float("Inf")
         
    def init_parameters_normal(self):
        # Weights and biases matrices (randomly initialized)
        self.W1 = np.random.randn(self.n_x, self.n_h)*0.1
        self.b1 = np.random.randn(1, self.n_h)*0.1
        self.W2 = np.random.randn(self.n_h, self.n_y)*0.1
        self.b2 = np.random.randn(1, self.n_y)*0.1

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))
    
    def forward(self, inputs):
        # Wx + b operation for the first layer
        Z1 = np.matmul(inputs, self.W1)
        Z1_b = Z1 + self.b1
        A1 = self.sigmoid(Z1_b)
        # Wx + b operation for the second layer
        Z2 = np.matmul(A1, self.W2)
        Z2_b = Z2 + self.b2
        y_pred = self.sigmoid(Z2_b)
        return y_pred
    
    def CE_loss(self, inputs, outputs):
        # CE loss function as before
        outputs_re = outputs.reshape(-1, 1)
        pred = self.forward(inputs)
        eps = 1e-10
        losses = outputs*np.log(pred + eps) + (1 - outputs)*np.log(1 - pred + eps)
        self.loss = -np.sum(losses)/outputs.shape[0]
        return self.loss
    
    def backward(self, inputs, outputs, alpha = 1e-5):
        # Get the number of samples in dataset
        m = inputs.shape[0]
        
        # Forward propagate
        Z1 = np.matmul(inputs, self.W1)
        Z1_b = Z1 + self.b1
        A1 = self.sigmoid(Z1_b)
        Z2 = np.matmul(A1, self.W2)
        Z2_b = Z2 + self.b2
        A2 = self.sigmoid(Z2_b)
    
        # Compute error term
        dL_dA2 = -outputs/A2 + (1 - outputs)/(1 - A2)
        dL_dZ2 = dL_dA2*A2*(1 - A2)
        dL_dA1 = np.dot(dL_dZ2, self.W2.T)
        dL_dZ1 = dL_dA1*A1*(1 - A1)
        
        # Gradient descent update rules
        self.W2 -= (1/m)*alpha*np.dot(A1.T, dL_dZ2)
        self.W1 -= (1/m)*alpha*np.dot(inputs.T, dL_dZ1)
        self.b2 -= (1/m)*alpha*np.sum(dL_dZ2, axis = 0, keepdims = True)
        self.b1 -= (1/m)*alpha*np.sum(dL_dZ1, axis = 0, keepdims = True)
        
        # Update loss
        self.CE_loss(inputs, outputs)
    
    def train(self, inputs, outputs, N_max = 1000, alpha = 1e-5, delta = 1e-5, display = True):
        # List of losses, starts with the current loss
        self.losses_list = [self.loss]
        # Repeat iterations
        for iteration_number in range(1, N_max + 1):
            # Backpropagate
            self.backward(inputs, outputs, alpha)
            new_loss = self.loss
            # Update losses list
            self.losses_list.append(new_loss)
            # Display
            if(display and iteration_number % (N_max*0.05) == 1):
                print("Iteration {} - Loss = {}".format(iteration_number, new_loss))
            # Check for delta value and early stop criterion
            difference = abs(self.losses_list[-1] - self.losses_list[-2])
            if(difference < delta):
                if(display):
                    print("Stopping early - loss evolution was less than delta on iteration {}.".format(iteration_number))
                break
        else:
            # Else on for loop will execute if break did not trigger
            if(display):
                print("Stopping - Maximal number of iterations reached.")
    
    def show_losses_over_training(self):
        # Initialize matplotlib
        fig, axs = plt.subplots(1, 2, figsize = (15, 5))
        axs[0].plot(list(range(len(self.losses_list))), self.losses_list)
        axs[0].set_xlabel("Iteration number")
        axs[0].set_ylabel("Loss")
        axs[1].plot(list(range(len(self.losses_list))), self.losses_list)
        axs[1].set_xlabel("Iteration number")
        axs[1].set_ylabel("Loss (in logarithmic scale)")
        axs[1].set_yscale("log")
        # Display
        plt.show()