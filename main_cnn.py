
"""
main_cnn.py
------------
Part 1: Build, train, and analyze a CNN from scratch on CIFAR-10.

Steps performed:
1. Load device info (GPU/CPU) via utils.py.
2. Load CIFAR-10 data via data_loader.py.
3. Initialize the SimpleCNN model from models/simple_cnn.py.
4. Define loss (CrossEntropy) and optimizer (Adam).
5. Train the model for 10–15 epochs, printing loss each epoch.
6. Evaluate test accuracy at the end of each epoch.
7. Plot training loss and test accuracy curves.
8. Visualize feature maps from the first convolutional layer.

Purpose:
Demonstrates supervised training of a custom CNN and
builds intuition for how filters learn visual patterns.
"""

import torch
from utils import get_device
from data_loader import get_dataloaders
from models.simple_cnn import SimpleCNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    device = get_device()
    trainloader, testloader = get_dataloaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 12                      # between 10–15
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # ---- evaluation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        test_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%")

    print("Training finished.")

    # ---- plot results ----
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("outputs/feature_maps", exist_ok=True)

    # Get one sample image
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    image = images[0].unsqueeze(0).to(device)  # take first image

    # ---- show and save original image ----
    def imshow(img, title=None, save_path=None):
        img = img / 2 + 0.5  # unnormalize back to [0,1]
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.show()

    imshow(images[0], title="Original CIFAR-10 Image",
           save_path="outputs/feature_maps/original_image.png")

    # ---- pass through first conv layer ----
    model.eval()
    with torch.no_grad():
        first_layer_output = model.conv1(image)
        feature_maps = first_layer_output.cpu().squeeze()  # shape [32, 32, 32]

    # ---- visualize and save 4 feature maps ----
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for i in range(4):
        axes[i].imshow(feature_maps[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Feature map {i + 1}")
    plt.tight_layout()
    plt.savefig("outputs/feature_maps/feature_maps_1to4.png", bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()
