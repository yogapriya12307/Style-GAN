# GANs and StyleGAN

## 📌 Overview

Generative Adversarial Networks (GANs) are deep learning models consisting of two neural networks: a **Generator** and a **Discriminator**. The generator creates synthetic data from random noise, while the discriminator distinguishes between real data and generated data. Both networks compete during training, helping the generator gradually improve the quality of its outputs.

This project explores different GAN architectures and implements a simplified StyleGAN-inspired model using PyTorch and the Fashion-MNIST dataset. The model generates synthetic grayscale fashion images and demonstrates important concepts such as latent-space interpolation and style mixing to understand how style information influences image generation.

## 🎯 Objectives

- Understand different GAN architectures and their working principles.
- Generate synthetic fashion images using a GAN-based model.
- Study the StyleGAN architecture and its style-based image-generation mechanism.
- Understand how the mapping network transforms random latent vectors into style representations.
- Perform latent-space interpolation to observe gradual changes between generated images.
- Explore style mixing to understand how style information from different inputs can influence generated images.

## 🧠 Types of GANs

- **Vanilla GAN:** The basic GAN architecture uses fully connected neural network layers. The generator creates synthetic data, and the discriminator identifies whether the input data is real or generated. It is commonly used to understand the fundamental concepts of adversarial learning.

- **Conditional GAN (cGAN):** Generates images based on additional information, such as class labels or specific conditions. For example, a model can be guided to generate a particular category of clothing by providing its label as input.

- **DCGAN:** Uses convolutional neural networks in the generator and discriminator to learn spatial features from images. It is widely used for image generation because convolutional layers help capture patterns such as shapes, edges, and textures.

- **SRGAN:** Focuses on generating high-resolution images from low-resolution inputs. It attempts to reconstruct finer details and improve image sharpness while maintaining the overall structure of the original image.

- **CycleGAN:** Performs image-to-image translation between two different visual domains without requiring paired training images. For example, it can transform photographs into paintings or convert images from one visual style to another.

- **StyleGAN:** Uses a mapping network and style-based control to influence image generation at different synthesis stages. It allows different levels of visual characteristics to be controlled through style information, making it suitable for controllable image synthesis.

- **LAPGAN:** Generates images progressively using a multi-resolution Laplacian pyramid. It starts with a low-resolution representation and gradually adds details at higher resolutions to construct the final image.

## 🎨 StyleGAN Architecture

StyleGAN is a generative model that provides greater control over the characteristics of generated images. It transforms a random latent vector into an intermediate style representation and uses this information to influence different stages of image synthesis.

**Main components:**

- **Mapping Network:** Converts the input latent vector \(z\) into an intermediate style vector \(w\) through a series of neural network layers. This representation helps control the characteristics of the generated image.

- **Learned Constant:** Provides a trainable initial feature representation for the synthesis network. Instead of directly starting image generation from the random latent vector, the network begins with this learned representation.

- **Synthesis Network:** Generates images using convolutional layers and upsampling operations. It progressively transforms feature maps into an image, with different stages contributing to the structure and details of the output.

- **AdaIN:** Adaptive Instance Normalization applies style-dependent scaling and bias to feature maps. This allows style information to influence the features learned at different stages of image generation.

- **Discriminator:** Evaluates whether an image is real or generated. It helps the generator improve its output through adversarial training by learning to distinguish synthetic images from genuine Fashion-MNIST images.

### Architecture Flow

```text
Latent Vector (z)
       |
       v
 Mapping Network
     z -> w
       |
       v
 Learned Constant
       |
       v
 Synthesis Blocks
 (Upsampling + Conv
      + AdaIN)
       |
       v
 Generated Image
       |
       v
 Discriminator
 (Real or Fake)
```

## 🧪 Experiments

### 1. Image Generation

The generator and discriminator are trained using the Fashion-MNIST dataset to produce synthetic grayscale clothing images. The generator takes random latent vectors as input and transforms them into images, while the discriminator learns to distinguish generated images from real clothing images.

During training, both networks are updated repeatedly. The generator gradually learns patterns from the dataset, such as clothing shapes and basic visual features. The generated images can be displayed using Matplotlib to observe the results of the training process.

### 2. Latent-Space Interpolation

Latent-space interpolation is used to observe how generated images change when moving between two different latent vectors. Two random latent vectors are selected, and intermediate vectors are created by gradually combining them.

These intermediate vectors are passed through the generator to produce a sequence of images. By observing the sequence, we can understand how changes in the latent representation influence the generated output and whether the transitions between images appear gradual.

### 3. Style Mixing

Style mixing demonstrates how style information from two different latent inputs can be combined during image generation. The style representation from one input can be used at certain synthesis stages, while the style representation from another input is applied at different stages.

This experiment illustrates how different style inputs can influence the generated image. In a simplified StyleGAN-inspired model, the effectiveness of style mixing depends on whether the synthesis network supports independent style control at different stages.

## ⚙️ Configuration

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Training images | 5,000 |
| Image size | 32 × 32 |
| Latent dimension | 32 |
| Batch size | 128 |
| Epochs | 10 |
| Optimizer | Adam |
| Framework | PyTorch |

The model uses 5,000 Fashion-MNIST training images and generates grayscale images at a resolution of 32 × 32 pixels. The latent dimension is set to 32, and a batch size of 128 is used during training. The model is trained for 10 epochs using the Adam optimizer.

These settings provide a small experimental setup for understanding GAN training and style-based image generation without requiring extensive computational resources.

## 🛠️ Technologies

- **Python:** Used to implement the model and manage the training process.
- **PyTorch:** Provides the neural network components, optimization methods, and training functionality.
- **Torchvision:** Used to access the Fashion-MNIST dataset and perform image transformations.
- **NumPy:** Supports numerical calculations and latent-vector operations.
- **Matplotlib:** Used to display generated images, interpolation results, style-mixing outputs, and training-loss graphs.
- **Jupyter Notebook:** Provides an interactive environment for executing the code and visualizing experimental results.

## 🚀 How to Run

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
cd GAN-StyleGAN
```

### Step 2: Install the required libraries

```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Step 3: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Run the project

Open `STYLEGAN.ipynb` in Jupyter Notebook and execute the cells in order. The notebook will load the dataset, initialize the generator and discriminator, train the model, and display the generated images and experimental visualizations, depending on the implemented code.

## 📊 Expected Results

- **Generated fashion images:** Synthetic grayscale clothing images resembling the patterns and shapes found in the Fashion-MNIST dataset.
- **Training-loss graphs:** Generator and discriminator loss plots that help observe the training process and changes in model behavior.
- **Latent-space interpolation visualizations:** A sequence of generated images showing how the output changes between two latent vectors.
- **Style-mixing visualizations:** Images demonstrating how style information from different latent inputs affects the generated output, provided the model supports stage-wise style control.

The quality of the results depends on the model architecture, dataset size, and number of training epochs. Since this is a simplified implementation, the generated images may contain limited details or unclear shapes.

## ⚠️ Limitations

This project implements a simplified StyleGAN-inspired model rather than the complete official StyleGAN architecture. The image resolution is limited to 32 × 32 pixels, and only 5,000 training images are used. Training for 10 epochs may not be sufficient to generate high-quality images.

GAN training can also be unstable, and the generator may produce similar-looking images or fail to capture all the variations present in the dataset. Furthermore, meaningful style mixing requires style information to be controlled at different synthesis stages, which may be limited in a simplified implementation.

## 🔮 Future Improvements

- Train the model for more epochs to improve the quality of generated images.
- Use larger datasets and higher image resolutions to capture more visual details.
- Compare the results with DCGAN and the official StyleGAN architecture.
- Improve the synthesis network to support more effective style mixing and latent-space control.
- Evaluate generated image quality using suitable quantitative metrics in addition to visual inspection.

## ✅ Conclusion

This project demonstrates the fundamental concepts of Generative Adversarial Networks and explores style-based image generation using a simplified StyleGAN-inspired model. By training on the Fashion-MNIST dataset, the project generates synthetic fashion images and examines the interaction between the generator and discriminator.

Latent-space interpolation demonstrates how changes in input vectors affect generated images, while style mixing explores how different style representations can influence image synthesis. Although the implementation has limitations in image resolution, dataset size, and training duration, it provides a foundation for understanding GAN architectures and the principles of style-based generation.

## 👨‍💻 Author

Yoga Priyaa A.V
