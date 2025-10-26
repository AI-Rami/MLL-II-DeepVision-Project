
"""
main_transfer.py
-----------------
Part 2: Transfer learning using a pretrained ResNet-18 model.

Steps performed:
1. Load CIFAR-10 and resize images to 224×224 (ResNet input size).
2. Load pretrained ResNet-18 (ImageNet weights) from torchvision.
3. Freeze all convolutional layers so only the final layer trains.
4. Replace the last fully-connected layer with 10-class output.
5. Train the new head for 12 epochs using GPU acceleration.
6. Evaluate accuracy each epoch.
7. Plot training loss and test accuracy curves.

Purpose:
Show how transfer learning reuses knowledge from ImageNet
to reach higher accuracy on CIFAR-10 with fewer training steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import get_device

def main():
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- Data prep ----
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # ---- Load pretrained ResNet18 ----
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # freeze all convolutional layers

    # Replace classifier head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    # ---- Loss & optimizer ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # ---- Training loop ----
    epochs = 12
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # ---- Evaluation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        test_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%")

    print("Transfer learning finished.")
    print("Model device check:", next(model.parameters()).device)

    # ---- Plot accuracy and loss ----
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), test_accuracies, label="Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
