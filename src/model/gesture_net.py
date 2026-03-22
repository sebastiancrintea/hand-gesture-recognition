import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureNet(nn.Module):
    def __init__(self, input_size: int = 42, hidden_size: int = 96, num_classes: int = 4) -> None:
        super(GestureNet, self).__init__()

        # 1. Input Layer -> Hidden Layer 1
        # We take 42 inputs (21 landmarks * 2 coords) and map them to 20 neurons
        self.fc1 = nn.Linear(input_size, hidden_size)

        # 2. Hidden Layer 1 -> Hidden Layer 2
        # Adding another layer allows the model to learn more complex patterns
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 3. Hidden Layer 2 -> Output Layer
        # Map parameters to the number of classes (e.g., 0, 1, 2, 3...)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # Dropout helps prevent overfitting (learning the training data too well)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the "Forward Pass" - how data flows through the network.
        """
        # Pass through first layer and apply ReLU (activation function)
        # ReLU turns negative numbers to 0 (adds non-linearity)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        # Note: We don't apply Softmax here because PyTorch's CrossEntropyLoss
        # calculates it automatically during training.
        x = self.fc3(x)

        return x
