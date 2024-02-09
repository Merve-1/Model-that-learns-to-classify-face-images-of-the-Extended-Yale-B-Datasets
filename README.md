# Extended YaleB Face Recognition

This repository contains code for face recognition using both Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) models. The models are implemented using TensorFlow and Keras libraries. Additionally, data augmentation is applied to enhance the training process.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed in your Python environment:

- `numpy`
- `cv2`
- `os`
- `google.colab`
- `tensorflow`
- `sklearn`
- `pandas`
- `tabulate`

You can install these libraries using the following command:

```bash
pip install numpy opencv-python google-colab tensorflow scikit-learn pandas tabulate
```

### Dataset

The code uses the Extended YaleB face dataset. Before running the code, make sure to download the dataset and place it in your Google Drive at the following path: `/content/drive/My Drive/ExtendedYaleB`.

## Code Structure

- **Configuration**: Import necessary libraries and mount Google Drive.

- **Loading Dataset**: Load images from the Extended YaleB dataset, resize them to 64x64 pixels, and prepare the data for training.

- **Splitting**: Split the dataset into training (80%), validation (10%), and testing (10%) sets. Normalize pixel values to be between [0, 1].

- **MLP Building**: Build a Multilayer Perceptron (MLP) model with three dense layers.

- **CNN Building**: Build a Convolutional Neural Network (CNN) model with convolutional and pooling layers.

- **Set Loss Function & Optimizer**: Compile both models using Sparse Categorical Crossentropy loss and Adam optimizer.

- **Train Model**: Train both models on the training set and validate on the validation set.

- **Evaluate Model**: Evaluate the models on the test set and print accuracy.

- **Data Augmentation**: Implement data augmentation using random horizontal flip, rotation, translation, and contrast adjustments.

- **MLP With Data Augmentation**: Build an MLP model with data augmentation and train/test the model.

- **CNN With Data Augmentation**: Build a CNN model with data augmentation and train/test the model.

## Bonus

Data augmentation is implemented using various transformations to improve the model's generalization.
