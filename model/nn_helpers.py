### Helper functions for the amino NN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to train the amino acid network
def train_amino_acid_network(network, data_loader, optimizer, criterion, batch_size, batch_logging=100):
    network.train()
    avg_loss = 0
    num_batches = 0
    for batch, (input_data, target_output) in enumerate(data_loader):
        optimizer.zero_grad()
        target_output = target_output.float()
        prediction = network(input_data)
        prediction = prediction.view(batch_size).clone()

        loss = criterion(prediction, target_output)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        num_batches += 1

        if (batch + 1) % batch_logging == 0:
            print('Batch [%d/%d], Train Loss: %.4f' % (batch + 1, len(data_loader.dataset) / len(target_output),
                                                       avg_loss / num_batches))

    return avg_loss / num_batches

# Function to test the amino acid network
def test_amino_acid_network(network, data_loader, criterion, batch_size):
    network.eval()
    test_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = network(data)
            output = output.view(batch_size).clone()
            output = output.float()
            target = target.float()
            test_loss += criterion(output, target).item()
            num_batches += 1

    test_loss /= num_batches
    return test_loss

# Function to log training results
def log_results(epoch, num_epochs, train_loss, train_loss_history, test_loss, test_loss_history, epoch_counter,
                print_interval=100):
    if (epoch % print_interval == 0):
        print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f' % (epoch + 1, num_epochs, train_loss, test_loss))
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    epoch_counter.append(epoch)

# Function to plot the loss graph
def graph_loss(epoch_counter, train_loss_hist, test_loss_hist, loss_name="Loss", start=0):
    fig = plt.figure()
    plt.plot(epoch_counter[start:], train_loss_hist[start:], color='blue')
    plt.plot(epoch_counter[start:], test_loss_hist[start:], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('#Epochs')
    plt.ylabel(loss_name)
    plt.show()

# Function to train and graph the amino acid network
def train_and_graph_amino_acid_network(network, training_loader, testing_loader, criterion, optimizer, num_epochs,
                                       learning_rate, batch_size, logging_interval=1):
    # Arrays to store training history
    test_loss_history = []
    epoch_counter = []
    train_loss_history = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        avg_loss = train_amino_acid_network(network, training_loader, optimizer, criterion, batch_size)
        test_loss = test_amino_acid_network(network, testing_loader, criterion, batch_size)
        log_results(epoch, num_epochs, avg_loss, train_loss_history, test_loss, test_loss_history, epoch_counter,
                    logging_interval)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(network.state_dict(), 'best_amino_acid_model.pt')

    # graph_loss(epoch_counter, train_loss_history, test_loss_history)


# Function to compute label accuracy for amino acid network
# def compute_label_accuracy_amino_acid(network, data_loader, label_text=""):
#     # Number correct, number of samples
#     numCorrect = 0
#     total_samples = 0 
#     network.eval()
#     with torch.no_grad():
#         for data, target in data_loader:
#             output = network(data)
#             output = output.view(100).clone()
#             x = (output == target).sum()
#             numCorrect += x.item()
#             total_samples += target.size(0)  # Increment the total_samples count
#             print('tot sample:', total_samples)
#             print('corr: ', numCorrect)
#     accuracy_percentage = 100. * numCorrect / total_samples  # Calculate the percentage of correct guesses
#     print('\n{}: Accuracy: {}/{} ({:.1f}%)'.format(
#         label_text, numCorrect, total_samples, accuracy_percentage))

# Function to compute label accuracy for amino acid network (for regression)
def compute_label_accuracy_amino_acid(network, data_loader, label_text=""):
    total_absolute_error = 0
    total_samples = 0

    network.eval()
    with torch.no_grad():
        for data, target in data_loader:
            output = network(data)
            output = output.view(-1).clone()
            
            absolute_error = torch.abs(output - target)
            total_absolute_error += absolute_error.sum().item()
            total_samples += target.size(0)

    mean_absolute_error = total_absolute_error / total_samples
    print('\n{}: Mean Absolute Error: {:.4f}'.format(label_text, mean_absolute_error))


# Function to draw predictions for amino acid network
def draw_predictions_amino_acid(network, dataset, num_rows=6, num_cols=10, skip_batches=0):
    data_generator = DataLoader(dataset, batch_size=num_rows * num_cols)
    #print('data generator:', data_generator)
    data_enumerator = enumerate(data_generator)

    for i in range(skip_batches):
        _, (input_data, target_output) = next(data_enumerator)

    _, (input_data, target_output) = next(data_enumerator)

    with torch.no_grad():
        predictions = network(input_data)
        pred_labels = predictions.argmax(dim=1)

    for row in range(num_rows):
        fig = plt.figure(figsize=(num_cols + 6, 5))
        for i in range(num_cols):
            plt.subplot(1, num_cols, i + 1)
            cur = i + row * num_cols
            draw_color = 'black' if pred_labels[cur].item() == target_output[cur].item() else 'red'
            plt.title("Prediction: {}, Actual: {}".format(pred_labels[cur].item(), target_output[cur].item()),
                      color=draw_color)
            plt.xticks([])
            plt.yticks([])

    plt.show()
