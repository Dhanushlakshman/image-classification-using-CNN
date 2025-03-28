# Convolutional Neural Network (CNN) Image Classifier
  This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into two categories: Cats and Dogs. The model is trained on labeled image datasets and evaluated for accuracy. It also includes image augmentation techniques to improve generalization.

  # CNN/
│── training_set/          # Training dataset
│── test_set/              # Test dataset
│── single_prediction/     # Folder for single image prediction
│── cnn_classifier.py      # Main script for training and testing
│── README.md              # Project documentation

# Features

* Data Preprocessing: Uses ImageDataGenerator to rescale and augment images.

* CNN Architecture: Includes convolutional, pooling, and dense layers.

* Binary Classification: Predicts whether an image is a cat or a dog.

* Performance Evaluation: Visualizes model accuracy and loss.

* Single Image Prediction: Predicts the class of an unseen image.

# Dependencies
pip install tensorflow, matplotlib, numpy, keras.

# Model Architecture

* The CNN consists of the following layers:

* Convolutional Layer (Conv2D) – Extracts features from input images.

* MaxPooling Layer (MaxPool2D) – Reduces feature map dimensions.

* Flatten Layer – Converts feature maps into a 1D vector.

* Fully Connected Layers (Dense) – Learns complex patterns.

* Output Layer (Sigmoid Activation) – Provides binary classification output.

 # Accuracy & Loss Graphs

* Accuracy Graph: Visualizes training and validation accuracy.

* Loss Graph: Tracks how well the model is learning.

# How to Run
* python imageclassification.ipnb

