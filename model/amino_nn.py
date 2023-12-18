import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class AminoAcidNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5, kernel_size=5):
        super(AminoAcidNN, self).__init__()
        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            #self.hidden_layers.append(nn.Conv1d(prev_size, hidden_size, kernel_size=kernel_size))
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

        # Initialize weights
        self.apply(weights_init)

    def forward(self, x):
        # Explicitly cast the input tensor to float
        x = x.float()

        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Forward through output layer
        x = self.output_layer(x)

        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
