# Adversarial-Attack-Defence
## **Introduction**
According to the recent studies, the vulnerability of state of the art Neural Networks to adversarial input samples has increased drastically. Neural network is an intermediate path or technique by which a computer learns to perform tasks using Machine learning algorithms. Machine Learning and Artificial Intelligence model has become fundamental aspect of life, such as self-driving cars, smart home devices, so any vulnerability is a significant concern. The smallest input deviations can fool these extremely literal systems and deceive their users as well as administrator into precarious situations. This article proposes a defense algorithm which utilizes the combination of an auto- encoder and block-switching architecture. Auto-coder is intended to remove any perturbations found in input images whereas block switching method is used to make it more robust against White-box attack. Attack is planned using FGSM method.

This majorly focuses on static image input and defence architecture. Following are the characteristics of the this model:
Combination of following two models to effectively defend both Black box and White Box attack.
* **Denoising AutoEncoder**
* **Block Switching Method**

## Installation
**Clone Repo**
```
git clone https://github.com/shoryasethia/Adversarial-Attack-Defence.git
```
**Install requirements**
```
!pip install tensorflow
!pip install numpy
!pip install keras
!pip install sklearn
!pip install matplotlib
!pip install os
!pip install PIL
```
## Architecture
![Simple Architecture](https://github.com/shoryasethia/Adversarial-Attack-Defence/blob/main/images/Architecture.png)

## ML Models Used
```
MobileNet
VGG16
Auto-Encoder
```
## Adversarial Attack
* Paper : [Fast Gradient Sign Method(FGSM)](https://github.com/shoryasethia/Adversarial-Attack-Defence/tree/main/Papers/Goodfellow%20FGSM)
```
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
```
## Defence 
1. **Denoising Auto Encoder :** Auto-encoders can be used for Noise Filtering purpose. By feeding them noisy data as inputs and clean data as outputs, it’s possible to remove noise from the input image. This way, auto-encoders can serve as denoisers.
2. **Paper : [Block Switching](https://github.com/shoryasethia/Adversarial-Attack-Defence/tree/main/Papers/Block%20Switching) :** Switching block in this experiment consists of multiple channels. Each regular model is split into a lower part, containing all convolutional layer. lower parts are again combined to form single output providing parallel channels of block switching while the other parts are discarded. These models tend to have similar characteristics in terms of classification accuracy and robustness, yet different model parameters due to random initialization and stochasticity in the training process.

# Block Switching Model
## Phase 1: Training Sub-Models

### Initial Training
Started by training `4` instances of the `modified VGG16` separately. Each instance begins with different random weight initializations. These models are based on VGG16 architecture but have been adapted for CIFAR-10 dataset with custom layers.

- Modified VGG16 with customized layers for CIFAR-10.
- Each model instance trained independently with variations due to random initialization and stochastic training.

## Phase 2: Creating the Switching Block

### Splitting the Models
After training, each modified VGG16 model is split into two parts:
- **Lower Part**: Includes the initial convolutional layers and feature extraction components.
- **Upper Part**: Comprises the fully connected layers and classification head.

- **Discard Upper Parts**: Remove the upper parts of all trained modified VGG16 models.

### Forming Parallel Channels
- **Grouping Lower Parts**: The lower parts (initial convolutional layers) of these trained models are grouped together to form parallel channels.
- **Base of Switching Block**: These parallel channels serve as the base of the switching block.

### Connecting the Switching Block
- **Adding Common Upper Model**: Introduce a new, randomly initialized common upper model.
- **Switching Mechanism**: Connect the parallel channels (lower parts) to the common upper model. At runtime, only one parallel channel is active for processing any given input, introducing stochastic behavior.

## Phase 3: Fine-Tuning the Combined Model

### Retraining
- **Combined Model Setup**: The switching block (parallel channels + common upper model) is retrained on the CIFAR-10 dataset.
- **Accelerated Training**: Retraining is faster since the lower parts (parallel channels) are pre-trained.
- **Adaptation Learning**: The common upper model learns to adapt to inputs from various parallel channels, ensuring robust classification regardless of the active channel.

### Customization Notes
- **Model Customization**: VGG16 architecture modified for CIFAR-10 with tailored layers.
- **Sub-Model Creation**: Multiple modified VGG16 models created and subsequently split for parallel channel formation.

## Why This work?
**Defense Against Adversarial Attacks:** This setup acts as a defense against adversarial attacks because:
* **Gradient Space:** Adversarial attacks often rely on finding directions in the gradient space that lead to misclassification. By using multiple parallel channels with different initializations, the gradient space becomes non-uniform across different instances of the model.
* **Stochastic Behavior:** The random selection of channels during inference introduces stochasticity, making it difficult for attackers to craft adversarial examples that generalize across different instances of the model.

## Results
  
