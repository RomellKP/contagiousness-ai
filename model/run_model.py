import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from amino_nn import AminoAcidNN
import nn_helpers as nh

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Amino acid data preparation (replace this with your actual data)
amino_acid_data = np.random.rand(100, 20)  # Example: 100 samples, 20 features (amino acids)

# Assuming you have a target variable representing contagiousness
contagiousness_labels = np.random.randint(2, size=100)  # Binary labels for contagiousness

# Convert NumPy arrays to PyTorch tensors
amino_acid_tensor = torch.FloatTensor(amino_acid_data)
contagiousness_labels_tensor = torch.LongTensor(contagiousness_labels)

# Create a TensorDataset for amino acid data and labels
amino_acid_dataset = TensorDataset(amino_acid_tensor, contagiousness_labels_tensor)

# Define training parameters
num_epochs_amino_acid = 10
print_interval_amino_acid = 1
learning_rate_amino_acid = 0.001
batch_size_amino_acid = 10
input_size_amino_acid = 20
hidden_size_amino_acid = 10
output_size_amino_acid = 2

# Create the amino acid network
amino_acid_network = AminoAcidNN(input_size_amino_acid, hidden_size_amino_acid, output_size_amino_acid)







### TODO:


# Set optimizer and loss



# Train, test, compute accuracy

nh.train_and_graph_amino_acid_network(network, training_loader, testing_loader, criterion, optimizer, num_epochs,
                                       learning_rate, logging_interval=1)

nh.test_amino_acid_network(network, data_loader, criterion)

nh.compute_label_accuracy_amino_acid(network, data_loader, label_text="")
