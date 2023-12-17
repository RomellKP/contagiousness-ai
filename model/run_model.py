import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from amino_nn import AminoAcidNN
import nn_helpers as nh

# Load data from the "processed_data.csv" file
df = pd.read_csv('../data/processed_data.csv')

# Extract X & y from dataframe
y = df['Contagiousness_Score'].values
df = df.drop(columns='Contagiousness_Score')
X = df.values

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Define training parameters
num_epochs_amino_acid = 10
print_interval_amino_acid = 1
learning_rate_amino_acid = 0.001
batch_size_amino_acid = 10
input_size_amino_acid = X_train.shape[1]  # Assuming the number of features is the second dimension
hidden_size_amino_acid = 10
output_size_amino_acid = 2

# Create the amino acid network
amino_acid_network = AminoAcidNN(input_size_amino_acid, hidden_size_amino_acid, output_size_amino_acid)

# Set optimizer and loss
optimizer = torch.optim.Adam(amino_acid_network.parameters(), lr=learning_rate_amino_acid)
criterion = torch.nn.CrossEntropyLoss()

# Create DataLoaders for training and testing sets
training_loader = DataLoader(train_dataset, batch_size=batch_size_amino_acid, shuffle=True)
testing_loader = DataLoader(test_dataset, batch_size=batch_size_amino_acid, shuffle=False)

# Train the amino acid network
nh.train_and_graph_amino_acid_network(amino_acid_network, training_loader, testing_loader, criterion, optimizer,
                                       num_epochs_amino_acid, learning_rate_amino_acid, logging_interval=1)

# Test the amino acid network
nh.test_amino_acid_network(amino_acid_network, testing_loader, criterion)

# Compute label accuracy for the amino acid network
nh.compute_label_accuracy_amino_acid(amino_acid_network, testing_loader, label_text="Contagiousness_Score")
