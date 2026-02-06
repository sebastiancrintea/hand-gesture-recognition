import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model.gesture_net import GestureNet
import os


def main():
    dataset_path = "model/keypoint_classifier/keypoint.csv"
    model_save_path = "model/keypoint_classifier/keypoint_classifier.pth"
    input_size = 42  # 21 landmarks * 2 (x, y)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading dataset...")
    # Read CSV. We assume no header, so first column is label, others are 42 coords
    df = pd.read_csv(dataset_path, header=None)

    X = df.iloc[:, 1:].values.astype("float32")  # Features (Landmarks)
    y = df.iloc[:, 0].values.astype("int64")  # Labels (Class IDs)

    num_classes = len(np.unique(y))
    print(f"Found {num_classes} classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    model = GestureNet(
        input_size=input_size, hidden_size=96, num_classes=num_classes
    ).to(device)

    # Loss Function (CrossEntropyLoss includes Softmax)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        optimizer.step()  # Update weights

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    onnx_save_path = model_save_path.replace(".pth", ".onnx")

    # Create dummy input for export trace (1 batch, 42 features)
    dummy_input = torch.randn(1, input_size, requires_grad=True).to(device)

    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_save_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_save_path}")


if __name__ == "__main__":
    main()
