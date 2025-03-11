# Lab 4: Experimenting with Modern CNN Architectures in PyTorch

## Deadline: March 18th at 2:30pm

## Summary

In this lab, you will explore, modify, and **compare two** pre-built Convolutional Neural Networks (CNNs) from PyTorch's `torchvision.models`. You will fine-tune these models on a dataset of your choice (e.g., CIFAR-10, TinyImageNet) and analyze their performance. You will also have the option to run your experiments on either a **CPU or GPU**.

**You are required to give a 3-5 minute presentation on March 18th at 2:30pm summarizing your findings.**

## Learning Outcomes

- Gain experience using **pre-trained CNN models** in PyTorch.
- Modify and fine-tune architectures for a **custom dataset**.
- **Compare different models** based on accuracy, loss, and computational efficiency.
- Experiment with model variations such as **depth, width, and activation functions**.
- Visualize learned features and discuss their impact on model performance.

## Resources
- For more on modern CNNs, read [d2l.ai's Modern CNN chapter](https://d2l.ai/chapter_convolutional-modern/index.html).
- For information on torchvision pre-trained models, see [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html).
- For datasets available in torchvision, see [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html).

## Step-by-Step Tasks

### 1. Load and Fine-tune Two Pre-built CNNs
- Load **two** pre-trained CNN architectures from `torchvision.models` (e.g., **ResNet-18 and DenseNet-121**).
    ```python
    import torchvision.models as models
    model1 = models.resnet18(pretrained=True)
    model2 = models.densenet121(pretrained=True)
    print(model1)
    print(model2)
    ```
- Modify the final classification layer of **each model** to match the dataset's number of classes.
- Fine-tune **both models** using transfer learning on a dataset of your choice.

### 2. Train and Validate the Models
1.  Load and preprocess the dataset (e.g., CIFAR-10) using `torchvision.transforms`.
2.  Split the dataset into training and validation sets.
3.  Define a loss function (e.g., `CrossEntropyLoss`) and an optimizer (e.g., `Adam`).
4.  Train **both models** for a specified number of epochs, printing the training loss for each epoch.
5.  After training, evaluate **both models** on the validation set and print the validation accuracy.

### 3. Visualize Model Features
1.  Extract the learned filters from the first convolutional layer of **each CNN**.
2.  Plot or display the filters in a grid to visualize the learned patterns for **each model**.
3.  **Compare the visualized filters** between the two models.

### 4 (Optional). Visualize Feature Maps and Interpret Results
- Use feature map visualization to analyze how **each model** interprets images.
- **Compare the feature maps** between the two models.

### 5. Presentation (March 18th at 2:30pm)

Prepare a 3-5 minute presentation summarizing your work. The presentation should include:

- **Overview of CNN Architectures**: Briefly describe the two CNN architectures you chose .
- **Experiment Setup**: Briefly describe the dataset, experiment variations, and hyperparameters used.
- **Results and Comparisons**: Present your key results, such as accuracy comparisons, loss trends, and feature map analysis between the two models.
- **Discussion**: Discuss any challenges faced and insights gained during the experiments.

#### Feature Visualization

Feature visualization helps you understand what patterns your CNN is detecting in images:

1. **Feature Maps**: These show what patterns activate specific neurons in your network.
   ```python
   # Register a hook to capture feature maps
   activations = {}
   def hook_fn(module, input, output):
       activations['features'] = output.detach()
   
   # Register the hook to the layer you want to visualize
   model.conv1.register_forward_hook(hook_fn)
   
   # Forward pass with your image
   with torch.no_grad():
       model(image)
       
   # Visualize the feature maps
   feature_maps = activations['features']
   ```

2. **Interpreting Visualizations**: Look for:
   - Early layers: detecting edges, textures, and simple patterns
   - Middle layers: more complex patterns like shapes and parts
   - Later layers: class-specific features

## Experiment with Model Variations (Choose at least two)

In addition to experimenting with hyperparameters, conduct at least two of the following experiments (can do different experiments on each model):

- **Option 1: Compare CNN Architectures** → Compare ResNet-18 vs. DenseNet-121.
- **Option 2: Change Network Depth** → For example, compare ResNet-18 vs. ResNet-50 or ResNet-34.
- **Option 3: Change Width/Channels** → Modify convolutional layers to have more or fewer channels.
- **Option 4: Change Activation Functions** → Replace ReLU with alternatives like LeakyReLU or Swish.
- **Option 5: Data Augmentation** → Train with and without augmentations (e.g., rotations, flips, color jitter).

    - **Depth:** Refers to the number of layers in the network. A deeper network has more layers.
    - **Width:** Refers to the number of channels in a convolutional layer or the number of neurons in a fully connected layer. A wider network has more channels/neurons.

## Example Code

The provided `src/main.py` demonstrates the core steps of this lab. In this example, a pre-trained model (like ResNet-18) is used first because it has already learned useful features from a large dataset (ImageNet). By fine-tuning it on our smaller dataset (e.g., CIFAR-10), we can achieve better performance with less training data and time. This technique is called **transfer learning**.

## Notes on CPU

Training on CPU is much slower (ResNet-18 might take 10x longer to train on CPU vs. GPU). You can run your program in Colab notebook using GPU while still including the final version of it in `src/main.py`. If you use CPU, consider doing the following:
- Reduce model size: if training is too slow, use a smaller model (e.g., `mobilenet_v2` instead of `resnet50`).
- Use fewer epochs: reduce training epochs to make it feasible.
- Smaller batch sizes: use `batch_size=16` or `batch_size=32`.

## Intermediate Checkpoints

- Push code to GitHub after each major step.
- Ensure you have regular commits demonstrating progress.

## Deliverables
1. **Python Script (`main.py`)** implementing the experiment.
2. **Report (`report.md`)** documenting the results of the experiments (see `report_template.md`).
3. A GitHub repository with consistent commit history.

- Clearly state whether you used a **CPU or GPU**, and discuss any differences in performance.

## Evaluation Criteria

This assignment uses contract-based grading. To receive `A` for the lab, you must:

- Implement the core CNN architectures, training loop, and validation for **two models** in `src/main.py`.
- Implement at least two of the experiment variations described above in `src/main.py`.
- Complete and submit your report in `writing/report.md`, including detailed results from your experiments and **comparisons between the two models**.
- Submit intermediate commits on GitHub.

For each component that is largely incomplete or incorrect, the assignment will receive a letter grade reduction.

GatorGrader is used to assess portions of the criteria above. The following will be checked by the GatorGrader tool:

1. **Source File Checks**:
    - Ensure that `main.py` file exists in the `src/` directory.
2. **Technical Writing Checks**:
    - Ensure that `report.md` file exists in the `writing/` directory.
    - Write a minimum number of meaningful words (300) in the report.
    - Complete all TODOs and remove the TODO markers in the `report.md`.
    - Ensure the report contains at least three visualizations of experimental graphs (comparing two models and two experiments).
3. **GitHub Repository Checks**:
    - Have at least a specific minimum number of commits (6) in the repository.

