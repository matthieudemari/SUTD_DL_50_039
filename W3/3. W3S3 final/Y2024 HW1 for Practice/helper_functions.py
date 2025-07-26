# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Torch
import torch
from torch.utils.data import TensorDataset, DataLoader


def val(min_val, max_val):
    return round(np.random.uniform(min_val, max_val), 2)


def class_for_val(val1, val2):
    k = np.pi
    return int(val2 >= -1/2 + 1/6*np.cos(val1*k)) + int(val2 >= 1/6 + 1/6*np.cos(val1*k*2))


def create_dataset(n_points, min_val, max_val):
    val1_list = np.array([val(min_val, max_val) for _ in range(n_points)], dtype = np.float32)
    val2_list = np.array([val(min_val, max_val) for _ in range(n_points)], dtype = np.float32)
    inputs = np.array([[v1, v2] for v1, v2 in zip(val1_list, val2_list)], dtype = np.float32)
    outputs = np.array([class_for_val(v1, v2) for v1, v2 in zip(val1_list, val2_list)], dtype = np.int64).reshape(n_points, 1)
    return val1_list, val2_list, inputs, outputs


def plot_dataset(min_val, max_val, train_val1_list, train_val2_list, train_outputs):
    # Initialize plot
    fig = plt.figure(figsize = (10, 7))

    # Scatter plot
    markers = {0: "x", 1: "o", 2: "P"}
    colors = {0: "r", 1: "g", 2: "b"}
    indexes_0 = np.where(train_outputs == 0)[0]
    v1_0 = train_val1_list[indexes_0]
    v2_0 = train_val2_list[indexes_0]
    indexes_1 = np.where(train_outputs == 1)[0]
    v1_1 = train_val1_list[indexes_1]
    v2_1 = train_val2_list[indexes_1]
    indexes_2 = np.where(train_outputs == 2)[0]
    v1_2 = train_val1_list[indexes_2]
    v2_2 = train_val2_list[indexes_2]
    plt.scatter(v1_0, v2_0, c = colors[0], marker = markers[0])
    plt.scatter(v1_1, v2_1, c = colors[1], marker = markers[1])
    plt.scatter(v1_2, v2_2, c = colors[2], marker = markers[2])

    # Display first true boundary
    x1 = [v1 for v1 in np.linspace(min_val, max_val, 50)]
    x2_true = [-1/2 + 1/6*np.cos(v1*np.pi) for v1 in x1]
    plt.plot(x1, x2_true, "k--", label = "True first boundary - Used in mock dataset generation")

    # Display second true boundary 
    x1 = [v1 for v1 in np.linspace(min_val, max_val, 50)]
    x2_true = [1/6 + 1/6*np.cos(v1*np.pi*2) for v1 in x1]
    plt.plot(x1, x2_true, "k", label = "True second boundary - Used in mock dataset generation")
    
    # Legend
    legend_elements = [Line2D([0], [0], marker = 's', color = 'w', label = 'Class 0 samples', \
                          markerfacecolor = 'r', markersize = 10),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'Class 1 samples', \
                          markerfacecolor = 'g', markersize = 10),
                       Line2D([0], [0], marker = 'P', color = 'w', label = 'Class 2 samples', \
                          markerfacecolor = 'b', markersize = 10),
                       Line2D([0], [0], color = 'k', linestyle = "--", label = 'Boundary 1'),
                       Line2D([0], [0], color = 'k', label = 'Boundary 2')]
    plt.legend(handles = legend_elements, loc = 'best')
    plt.xlabel("v1 value")
    plt.ylabel("v2 value")

    # Show
    plt.show()
    
    
def generate_loaders(min_val, max_val, n_points_train, n_points_test, n_points_valid, batch_size, device):
    # Generate dataset (train)
    np.random.seed(47)
    train_val1_list, train_val2_list, train_inputs, train_outputs = create_dataset(n_points_train, min_val, max_val)

    # Convert to tensors and send to device (CUDA or CPU)
    train_inputs_pt = torch.from_numpy(train_inputs).to(device)
    train_outputs_pt = torch.from_numpy(train_outputs).to(device)

    # Train dataloader
    train_loader = DataLoader(TensorDataset(train_inputs_pt, train_outputs_pt), \
                              batch_size = batch_size, \
                              shuffle = True)

    # Generate dataset (test)
    np.random.seed(17)
    test_val1_list, test_val2_list, test_inputs, test_outputs = create_dataset(n_points_test, min_val, max_val)

    # Convert to tensors and send to device (CUDA or CPU)
    test_inputs_pt = torch.from_numpy(test_inputs).to(device)
    test_outputs_pt = torch.from_numpy(test_outputs).to(device)

    # Test dataloader
    test_loader = DataLoader(TensorDataset(test_inputs_pt, test_outputs_pt), \
                             batch_size = n_points_test, \
                             shuffle = False)

    # Generate dataset (validation)
    np.random.seed(27)
    valid_val1_list, valid_val2_list, valid_inputs, valid_outputs = create_dataset(n_points_valid, min_val, max_val)

    # Convert to tensors and send to device (CUDA or CPU)
    valid_inputs_pt = torch.from_numpy(valid_inputs).to(device)
    valid_outputs_pt = torch.from_numpy(valid_outputs).to(device)

    # Test dataloader
    valid_loader = DataLoader(TensorDataset(valid_inputs_pt, valid_outputs_pt), \
                             batch_size = n_points_valid, \
                             shuffle = False)
    
    # Return
    return train_loader, test_loader, valid_loader


def display_train(epoch, num_epochs, i, model, correct, total, loss, train_loader, valid_loader, device):
    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
    train_accuracy = correct/total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}')
    valid_accuracy = eval_valid(model, valid_loader, epoch, num_epochs, device)
    return train_accuracy, valid_accuracy

    
    
def eval_valid(model, valid_loader, epoch, num_epochs, device):
    # Compute model train accuracy on test after all samples have been seen using test samples
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in valid_loader:
            # Get images and labels from test loader
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)

            # Forward pass and predict class using max
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Check if predicted class matches label and count numbler of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute final accuracy and display
    valid_accuracy = correct/total
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {valid_accuracy:.4f}')
    return valid_accuracy


def eval_test(model, test_loader, device):
    # Compute model test accuracy on test after training
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            # Get images and labels from test loader
            images = inputs.to(device)
            labels = labels.reshape(-1).to(device)

            # Forward pass and predict class using max
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Check if predicted class matches label and count numbler of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute final accuracy and display
    test_accuracy = correct/total
    print(f'Ended Training, Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy
