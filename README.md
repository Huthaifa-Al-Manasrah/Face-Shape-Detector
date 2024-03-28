# Face Shape Detector

## Description

Face Shape Detector is a comprehensive project that combines a Flutter mobile application for detecting and analyzing human face shapes with a Python-based machine learning model trained to recognize various face shapes.

The Flutter app provides users with an intuitive interface to upload images containing human faces. The uploaded images are then analyzed using an AI-powered model to determine the face shape, which is categorized into one of the following types: Diamond, Heart, Oblong, Oval, Round, Square, and Triangle.

The Python model, built using TensorFlow and Keras, was trained on a dataset of images representing different face shapes. It utilizes Convolutional Neural Networks (CNNs) to perform accurate and efficient face shape classification.

## Project Structure

The repository is organized into two main directories:

- **flutter_app:** Contains the source code for the Flutter mobile application.
- **model_training:** Contains the Python code and related files for training the machine learning model.

## Technologies Used

- Flutter: Cross-platform UI toolkit for building natively compiled applications.
- TensorFlow and Keras: Deep learning frameworks used for building and training the face shape classification model.
- OpenCV: Library used for image processing and manipulation.
- NumPy: Library used for numerical operations and array manipulations.

## Getting Started

To get started with the Face Shape Detector project, follow these steps:

1. Clone the repository to your local machine.
2. Set up and run the Flutter app for face shape detection using the instructions provided in the `flutter_app` directory.
3. Explore the `model_training` directory to learn more about the Python code used for training the face shape classification model.

## Contributing

Contributions to the Face Shape Detector project are welcome! Whether you're interested in improving the Flutter app, enhancing the machine learning model, or fixing bugs, feel free to fork the repository and submit pull requests.
