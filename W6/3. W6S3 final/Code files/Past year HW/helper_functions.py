# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def create_dataset(n_points, a, b, x0):
    times = np.linspace(0, n_points - 1, n_points)
    values = [x0]
    for t in range(n_points - 1):
        next_val = values[-1]*(1 + a) + values[-1]*b*np.random.randn()
        values.append(next_val)
    values = np.array(values)
    return times, values
    

def save_dataset(times, values, excel_file_path = 'dataset.xlsx'):
    # Create a DataFrame from these lists
    data = {'times': times, 'values': values}
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index = False)
    

def load_dataset(excel_file_path = 'dataset.xlsx'):
    df = pd.read_excel(excel_file_path)
    times = df['times'].values
    values = df['values'].values
    return times, values
    

def plot_dataset(times, values):
    # Initialize plot
    plt.figure(figsize = (10, 7))
    plt.plot(times, values)
    plt.show()
    

def visualize_samples(inputs, mid, outputs):
    plt.figure(figsize = (10, 7))
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    mid = mid.numpy()
    times1 = [i for i in range(len(inputs))]
    times2 = [len(inputs)]
    times3 = [len(inputs) + i + 1 for i in range(len(outputs))]
    plt.scatter(times1, inputs, label = "Inputs", c = "b")
    plt.scatter(times2, mid, label = "Mid", c = "g")
    plt.scatter(times3, outputs, label = "Outputs", c = "r")
    plt.plot(times1 + times2 + times3, np.hstack([inputs, mid, outputs]),  "k--")
    plt.legend(loc = "best")
    plt.show()
    

def test_model(model, dataloader, seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    # Draw sample from dataloader (reproducible thanks to seeding)
    inputs, outputs, mid = next(iter(dataloader))
    # Predict
    pred = model(inputs, outputs, mid)
    # Compute metrics
    print("Ground truth: ", outputs[0, :])
    print("Prediction: ", pred[0, :])
    print("Mean Square Error for Sample: ", np.mean((outputs[0, :].detach().numpy() - pred[0, :].detach().numpy())**2))
    
    
def visualize_some_predictions(model, dataloader):
    fig, axs = plt.subplots(2, 2, figsize = (15, 10))
    
    index1 = 2486
    ax = axs[0, 0]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128, :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label = "Inputs", c = "b")
    ax.scatter(times2, mid1, label = "Mid", c = "g")
    ax.scatter(times3, outputs1, label = "Outputs", c = "r")
    err = np.mean((outputs1 - pred1)**2)
    ax.scatter(times3, pred1, label = "Predictions - Error = {}".format(err), c = "c", marker = "x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]),  "k--")
    ax.legend(loc = "best")

    index1 = 2986
    ax = axs[0, 1]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128, :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label = "Inputs", c = "b")
    ax.scatter(times2, mid1, label = "Mid", c = "g")
    ax.scatter(times3, outputs1, label = "Outputs", c = "r")
    err = np.mean((outputs1 - pred1)**2)
    ax.scatter(times3, pred1, label = "Predictions - Error = {}".format(err), c = "c", marker = "x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]),  "k--")
    ax.legend(loc = "best")

    index1 = 3486
    ax = axs[1, 0]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128, :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label = "Inputs", c = "b")
    ax.scatter(times2, mid1, label = "Mid", c = "g")
    ax.scatter(times3, outputs1, label = "Outputs", c = "r")
    err = np.mean((outputs1 - pred1)**2)
    ax.scatter(times3, pred1, label = "Predictions - Error = {}".format(err), c = "c", marker = "x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]),  "k--")
    ax.legend(loc = "best")

    index1 = 3986
    ax = axs[1, 1]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128, :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label = "Inputs", c = "b")
    ax.scatter(times2, mid1, label = "Mid", c = "g")
    ax.scatter(times3, outputs1, label = "Outputs", c = "r")
    err = np.mean((outputs1 - pred1)**2)
    ax.scatter(times3, pred1, label = "Predictions - Error = {}".format(err), c = "c", marker = "x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]),  "k--")
    ax.legend(loc = "best")

    plt.tight_layout()
    plt.show()