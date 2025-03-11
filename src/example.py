import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Determine if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a uniform size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the CIFAR-10 dataset for training and validation
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Create data loaders to handle batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for CIFAR-10 (10 classes)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the GPU if available
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2  # Number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        # Move data to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the training loss for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Validation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for validation
    for images, labels in val_loader:
        # Move data to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the validation accuracy
print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), "resnet18_cifar10.pth")

# Visualize the first layer's learned filters
first_layer_weights = model.conv1.weight.data.cpu()
num_filters_to_display = 16
plt.figure(figsize=(8, 8))
for i in range(num_filters_to_display):
    plt.subplot(4, 4, i + 1)
    plt.imshow(first_layer_weights[i][0], cmap='gray')  # Display the first channel in grayscale
    plt.axis('off')
plt.suptitle('First Layer Convolutional Filters')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.show()

# Feature map visualization
def visualize_feature_maps(model, img, label):
    # Get the first convolutional layer
    layer = model.conv1
    
    # Forward, saving the output
    img_tensor = img.unsqueeze(0).to(device)
    output = layer(img_tensor)

    # Get prediction
    with torch.no_grad():
        model.eval()
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction).item()

    # Plot the feature maps
    n_feats = output.shape[1]
    fig = plt.figure(figsize=(12, 12))
    
    # Plot original image
    plt.subplot(4, 4, 1)
    img_np = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title(f"Original, Predicted: {predicted_class}, Label: {label}")
    plt.axis('off')
    
    # Plot feature maps
    for i in range(min(n_feats, 15)):
        plt.subplot(4, 4, i + 2)
        plt.imshow(output[0, i].data.cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get a sample image from the validation set
dataiter = iter(val_loader)
images, labels = next(dataiter)
img = images[0]
label = labels[0]

# Visualize feature maps
visualize_feature_maps(model, img, label)
