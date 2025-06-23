import torch.nn as nn

import torch.nn as nn

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "softplus":
        return nn.Softplus()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(LSTM, self).__init__()
        act = get_activation(activation)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)  # Fully connected layers
        self.act1 = act
        self.fc2 = nn.Linear(64, 32)
        self.act2 = act
        self.fc3 = nn.Linear(32, output_size)  # Maps to the target dimension

    def forward(self, x):
        # LSTM layer
        output, _ = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)

        # Process each time step through FC layers
        output = self.fc1(output)  # (batch_size, seq_length, 64)
        output = self.act1(output)
        output = self.fc2(output)  # (batch_size, seq_length, 32)
        output = self.act2(output)
        output = self.fc3(output)  # (batch_size, seq_length, output_size)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        super(MLP, self).__init__()
        act = get_activation(activation)

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(act)
            in_features = h

        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)