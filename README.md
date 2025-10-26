Machine Learning II – Compulsory 2
Deep Vision: CNN from Scratch and Transfer Learning

This project is part of the course Machine Learning II (INN University).
It demonstrates image classification on the CIFAR-10 dataset using two methods:

A custom Convolutional Neural Network (CNN) trained from scratch.

A pretrained ResNet18 model using transfer learning.

Both approaches are trained, evaluated, and compared on accuracy and training behavior.

Project Structure

utils.py

Detects if a CUDA GPU is available and returns the correct device.

Keeps all other scripts clean and hardware independent.

data_loader.py

Downloads and normalizes the CIFAR-10 dataset.

Converts images to tensors and scales pixel values to [-1, 1].

Creates DataLoader objects for training and testing.

models/simple_cnn.py

Defines the CNN architecture used in Part 1 (training from scratch).

Contains 3 convolutional blocks with BatchNorm, ReLU, and MaxPooling.

Ends with two fully connected layers and dropout for regularization.

main_cnn.py

Trains the custom CNN from scratch.

Loads data, defines optimizer and loss, and trains for 10–15 epochs.

Plots training loss and test accuracy.

Visualizes 4 feature maps from the first convolution layer.

main_transfer.py

Performs transfer learning using pretrained ResNet18.

Freezes all convolutional layers and replaces the last layer with a 10-class output.

Trains only the new head for 12 epochs.

Plots training loss and test accuracy.

outputs/feature_maps/

Contains the original image and the 4 feature maps produced by the first CNN layer.

Part 1 – CNN From Scratch

Dataset: CIFAR-10 (32x32 RGB images, 10 classes)
Training setup:

12 epochs

Batch size 64

Optimizer: Adam (learning rate 0.001)

Loss function: CrossEntropyLoss

Output:

Training loss and test accuracy plots.

Feature map visualization from the first convolutional layer.

Expected accuracy: around 78%.

Part 2 – Transfer Learning

Model: Pretrained ResNet18 (ImageNet weights)
Training setup:

Input images resized to 224x224.

Frozen convolutional base.

New fully connected layer with 10 outputs.

12 training epochs.

Output:

Training loss and test accuracy plots.

Expected accuracy: around 80–85%.

How to Run

To train the CNN from scratch:
python main_cnn.py

To train the transfer learning model:
python main_transfer.py

Summary

CNN from scratch: about 78% accuracy after 12 epochs.
ResNet18 transfer learning: about 81% accuracy after 12 epochs.

Learning Outcomes

Understand how CNNs learn to detect visual patterns.

Observe the difference between learning from scratch and transfer learning.

Learn how to organize ML projects with clear structure and modular code.

This completes Compulsory 2 for Machine Learning II.
Project verified to work with Python 3.12
