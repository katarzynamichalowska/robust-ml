import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)  # Fully connected layers
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)  # Maps to the target dimension

    def forward(self, x):
        # LSTM layer
        output, _ = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)

        # Process each time step through FC layers
        output = self.fc1(output)  # (batch_size, seq_length, 64)
        output = self.relu1(output)
        output = self.fc2(output)  # (batch_size, seq_length, 32)
        output = self.relu2(output)
        output = self.fc3(output)  # (batch_size, seq_length, output_size)
        return output.squeeze(-1)  # Remove last dimension if output_size = 1
