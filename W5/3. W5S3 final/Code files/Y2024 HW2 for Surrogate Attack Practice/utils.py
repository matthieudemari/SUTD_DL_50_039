# Future
from __future__ import print_function
# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Torchvision
from torchvision import datasets, transforms


'''
Test attack on original model (unprotected)
'''
def test_attack(original_model, iugm_attack, device, test_loader, epsilon, max_iter = 10):

    # Counter for correct values (used for accuracy)
    correct_counter = 0
    
    # List of successful adversarial samples
    adv_examples_list = []

    # Loop over all examples in test set
    for image, label in test_loader:
        
        # Send the data and label to the device
        image, label = image.to(device), label.to(device)

        # Set requires_grad attribute of tensor to force torch to
        # keep track of the gradients of the image
        # (Needed for the ugm_attack() function!)
        image.requires_grad = True

        # Pass the image through the model
        output = original_model(image)
        # Get the index of the max log-probability
        _, init_pred = torch.max(output.data, 1)

        # If the initial prediction is wrong, do not bother attacking, skip current image
        if init_pred.item() != label.item():
            continue
            
        # Call IUGM Attack
        eps_image = iugm_attack(image, epsilon, original_model, label, max_iter)

        # Re-classify the epsilon image
        output2 = original_model(eps_image)
        # Get the index of the max log-probability
        _, eps_pred = torch.max(output2.data, 1)

        # Check for successful attack
        # (Successful meaning eps_pred label different from init_pred)
        if eps_pred.item() == label.item():
            correct_counter += 1
            # Special case for saving 0 epsilon examples
            # (Maximal number of saved samples is set to 5)
            if (epsilon == 0) and (len(adv_examples_list) < 5):
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                adv_examples_list.append((init_pred.item(), eps_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            # (Maximal number of saved samples is set to 5)
            if len(adv_examples_list) < 5:
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                adv_examples_list.append((init_pred.item(), eps_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon value
    final_acc = correct_counter/float(len(test_loader))
    
    # Display for progress
    print("Epsilon: {} - Model Accuracy (under attack) = {}/{} = {}".format(epsilon, \
                                                                            correct_counter, \
                                                                            len(test_loader), \
                                                                            final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples_list


'''
Run test attack on original model (unprotected) for several epsilon values
'''    
def run_attacks_for_epsilon(epsilons, original_model, iugm_attack, device, test_loader, max_iter):
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = test_attack(original_model, iugm_attack, device, test_loader, eps, max_iter)
        accuracies.append(acc)
        examples.append(ex)
    return accuracies, examples


'''
Display attack curves for original model (unprotected)
'''
def display_attack_curves(epsilons, epsilons2, accuracies, accuracies2):
    # Initialize figure
    plt.figure(figsize = (10, 7))

    # Display accuracy vs. Epsilon values plot
    plt.plot(epsilons, accuracies, "o-", color = "red", label = "One-shot attack on pre-trained model")
    plt.plot(epsilons2, accuracies2, "o-", color = "blue", label = "Iterated Attack on pre-trained model")

    # Adjust x-axis and y-axis labels and ticks
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.title("Accuracy vs. Epsilon value (one-shot as red, iterated as blue)")
    plt.xlabel("Epsilon value")
    plt.ylabel("Accuracy")
    plt.legend(loc = "best")

    # Display
    plt.show()
    


'''
Display adversarial samples
'''
def display_adv_samples(epsilons, examples):
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0

    # Initialize figure
    plt.figure(figsize = (20, 20))

    # Browse through epsilon values and adversarial examples
    for i in range(len(epsilons)):
        # If example list does not contain 5 samples for this epsilon,
        # do not display this epsilon value
        if len(examples[i]) != 5:
            print("Skipped value eps = {}, did not have 5 samples to display".format(epsilons[i]))
            continue
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)

            # Remove x-axis and y-axis ticks from plot
            plt.xticks([], [])
            plt.yticks([], [])

            # Labels for y axis
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize = 14)

            # Labels for each image subplot
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))

            # Reshape ex for imshow
            ex = np.transpose(ex, (1, 2, 0))

            # Display image
            plt.imshow(ex)

    # Display full plot
    plt.tight_layout()
    plt.show()
    
    
'''
Display attack curves for surrogate model attack (pretending original model is protected)
'''
def display_attack_curves_surr(epsilons, epsilons2, \
                               accuracies, accuracies2, \
                               epsilons_surr, accuracies_surr, \
                               epsilons_surr2, accuracies_surr2):
    # Initialize figure
    plt.figure(figsize = (10, 7))

    # Display accuracy vs. Epsilon values plot
    plt.plot(epsilons, accuracies, "o-", color = "red", label = "One-shot attack on pre-trained model")
    plt.plot(epsilons2, accuracies2, "o-", color = "blue", label = "Iterated Attack on pre-trained model")
    plt.plot(epsilons_surr, accuracies_surr, "o-", color = "green", label = "One-shot attack via surrogate model")
    plt.plot(epsilons_surr2, accuracies_surr2, "o-", color = "black", label = "Iterated Attack via surrogate model")

    # Adjust x-axis and y-axis labels and ticks
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.title("Accuracy vs. Epsilon value (one-shot as red, iterated as blue)")
    plt.xlabel("Epsilon value")
    plt.ylabel("Accuracy")
    plt.legend(loc = "best")

    # Display
    plt.show()
    
    
def test_attack_surr(original_model, surrogate_model, iugm_attack_surr, device, test_loader, epsilon, max_iter = 10):

    # Counter for correct values (used for accuracy)
    correct_counter = 0
    
    # List of successful adversarial samples
    adv_examples_list = []

    # Loop over all examples in test set
    for image, label in test_loader:
        
        # Send the data and label to the device
        image, label = image.to(device), label.to(device)

        # Set requires_grad attribute of tensor to force torch to
        # keep track of the gradients of the image
        # (Needed for the ugm_attack() function!)
        image.requires_grad = True

        # Pass the image through the model
        output = original_model(image)
        # Get the index of the max log-probability
        _, init_pred = torch.max(output.data, 1)

        # If the initial prediction is wrong, do not bother attacking, skip current image
        if init_pred.item() != label.item():
            continue
            
        # Call IUGM Attack
        eps_image = iugm_attack_surr(image, epsilon, original_model, surrogate_model, label, max_iter)

        # Re-classify the epsilon image
        output2 = original_model(eps_image)
        # Get the index of the max log-probability
        _, eps_pred = torch.max(output2.data, 1)

        # Check for successful attack
        # (Successful meaning eps_pred label different from init_pred)
        if eps_pred.item() == label.item():
            correct_counter += 1
            # Special case for saving 0 epsilon examples
            # (Maximal number of saved samples is set to 5)
            if (epsilon == 0) and (len(adv_examples_list) < 5):
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                adv_examples_list.append((init_pred.item(), eps_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            # (Maximal number of saved samples is set to 5)
            if len(adv_examples_list) < 5:
                adv_ex = eps_image.squeeze().detach().cpu().numpy()
                adv_examples_list.append((init_pred.item(), eps_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon value
    final_acc = correct_counter/float(len(test_loader))
    
    # Display for progress
    print("Epsilon: {} - Model Accuracy (under attack) = {}/{} = {}".format(epsilon, \
                                                                            correct_counter, \
                                                                            len(test_loader), \
                                                                            final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples_list


'''
Run test attack on surrogate model (assuming original model is protected) for several epsilon values
'''
def run_attacks_for_epsilon_surr(epsilons, original_model, surrogate_model, iugm_attack_surr, \
                                 device, test_loader, max_iter):
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = test_attack_surr(original_model, surrogate_model, iugm_attack_surr, device, test_loader, eps, max_iter)
        accuracies.append(acc)
        examples.append(ex)
    return accuracies, examples