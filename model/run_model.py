### IMPORTS #############################################
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from amino_nn import AminoAcidNN
import nn_helpers as nh
from imblearn.over_sampling import SMOTE
#########################################################



######## GET DATA ######################################

# Load prepped data from files 
df1 = pd.read_csv('../data/processed_data.csv')
df2 = pd.read_csv('../data/processed_data2.csv')

# Combine dfs
df = pd.concat([df1, df2], ignore_index=True)


# Extract X & y from dataframe
y = df['Contagiousness_Score'].values
df = df.drop(columns=['Contagiousness_Score', 'Collection_Date', 'Accession', 'Pangolin'])
all_X = df.values

# Sample for balanced regression outputs
all_X_np = np.array(all_X)
y_np = np.array(y)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(all_X_np, y_np)


# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(all_X, y, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# Use StandardScaler to ensure inputs are between 0 and 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train_scaled)
X_test_tensor = torch.from_numpy(X_test_scaled)

# Create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#########################################################



######### HYPERPARAMS ####################################

epochs = 300
learning_rate = 0.0005
batch_size = 50
logging_interval = 10
input_size = X_train.shape[1]  # 21 
hidden_sizes = [256, 128, 64]
output_size = 1

#########################################################


# Create the amino acid network
model = AminoAcidNN(input_size, hidden_sizes, output_size)
# Set optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()

# Create DataLoaders for training and testing sets
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the amino acid network
print("Training...")
nh.train_and_graph_amino_acid_network(model, training_loader, testing_loader, criterion, optimizer,
                                       epochs, learning_rate, batch_size, logging_interval=logging_interval)

# Test the amino acid network
print("Testing...")
nh.test_amino_acid_network(model, testing_loader, criterion, batch_size)

# Compute label accuracy for the amino acid network
nh.compute_label_accuracy_amino_acid(model, testing_loader, label_text="Contagiousness_Score")
