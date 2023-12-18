import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Arch amino acid counts
class AminoAcidNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AminoAcidNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = x.float()
        x = self.fc2(x)
        return F.sigmoid(x)
        #return x
