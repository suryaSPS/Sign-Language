# Sign Language Detection Project Report
![alt text](https://github.com/AryanDahiya00/Sign-Language-Detection/blob/main/images/dataset_classes.png)
## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Possible Approaches](#possible-approaches)
4. [Methodology](#methodology)
   - [Data Acquisition and Preprocessing](#data-acquisition-and-preprocessing)
   - [CNN Model Architecture](#cnn-model-architecture)
   - [Evaluation](#evaluation)
5. [Results and Discussion](#results-and-discussion)
   - [Results](#results)
   - [Discussion](#discussion)
6. [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

This project addresses the communication challenges faced by Deaf and hard-of-hearing (DHH) individuals who rely on Sign Language. The scarcity of qualified sign language interpreters creates barriers in crucial settings like healthcare, legal proceedings, and educational institutions. To bridge this gap, we propose a groundbreaking solution: a real-time sign language gesture recognition system powered by machine learning, specifically using Convolutional Neural Networks (CNNs).

The system aims to revolutionize communication accessibility for sign language users by analyzing video frames to accurately identify sign language gestures. This technology has the potential to facilitate seamless communication across diverse social and professional settings, empowering DHH individuals to participate fully in society.

## Problem Statement

The primary challenges addressed by this project include:

1. Limited availability of qualified sign language interpreters
2. Communication barriers in critical settings (healthcare, legal, education)
3. Potential misunderstandings and delays in accessing vital services
4. Restricted participation of DHH individuals in society due to communication gaps

The project aims to develop a solution that can overcome these challenges and provide real-time sign language interpretation.

## Possible Approaches

Three approaches were explored for the Sign Language Gesture Recognition System:

1. **MEDIAPIPE**
   - Goal: Utilize MediaPipe, an open-source framework by Google, to identify and track hand landmarks in video frames.
   - Problem: Camera quality significantly impacted the accuracy of hand landmark detection and storage. Poor resolution or lighting conditions hindered reliable tracking.

2. **OBJECT DETECTION**
   - Goal: Implement object detection algorithms to identify hands within video frames and potentially classify them based on their position and shape.
   - Problem: This approach might be overly complex for the specific task. Object detection algorithms are designed for a wider range of objects, and customizing them for hand gestures only might introduce unnecessary processing overhead.

3. **CONVOLUTIONAL NEURAL NETWORK (CNN) ARCHITECTURE**
   - Goal: Develop and implement a CNN model specifically designed for sign language gesture recognition.
   - Implementation: Built a model with CNN layers to extract features from frames containing hand gestures. This involved convolutional layers followed by pooling layers to reduce data size and activation functions to introduce non-linearity.

The CNN approach was chosen as the most suitable for this project due to its effectiveness in extracting features from images and videos.

## Methodology

### Data Acquisition and Preprocessing

- Dataset: American Sign Language (ASL) gestures from Kaggle
- Format: CSV files containing pixel values of images
- Training set: 27,455 images
- Test set: 7,172 images
- Image representation: 784 pixels (28x28) in grayscale
- Labels: 0 to 25 representing letters A-Z (excluding J and Z)

Preprocessing steps:
1. Resizing images to 64x64 pixels
2. Converting to float type
3. Normalizing pixel values between 0 and 1
4. Applying data augmentation techniques to improve model robustness

### CNN Model Architecture

The CNN model for ASL gesture classification was designed with multiple layers optimized for image recognition:

1. Convolutional layers for feature extraction
2. Batch normalization layers for improved training stability
3. Max-pooling layers for spatial down-sampling
4. Dropout layers to prevent overfitting
5. Dense layers for final classification
6. Output layer with softmax activation function for 24 gesture classes

The model was compiled using:
- Optimizer: Adam (for dynamic learning rate adjustment)
- Loss function: Categorical cross-entropy (for multi-class classification)

### Evaluation

Performance evaluation metrics included:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Training and validation accuracy/loss curves

A function was defined to predict random images, where the input image is processed, and the predicted alphabet is printed.

## Results and Discussion

### Results

1. **Accuracy**: The model achieved a high accuracy of 96.35% on the test set.

2. **Training and Validation Curves**: 
   - Showed consistent performance without significant overfitting
   - Indicated good generalization to unseen data

3. **Confusion Matrix**: 
   - Confirmed robust performance across all classes
   - Most gestures were accurately recognized

4. **Gesture Recognition Function**:
   - Successfully preprocessed input images
   - Utilized the trained CNN model for classification
   - Demonstrated high accuracy in predicting ASL alphabets

### Discussion

1. **Model Efficacy**: High accuracy and balanced precision, recall, and F1-scores indicated effective learning of ASL gestures.

2. **Regularization Techniques**: Dropout and batch normalization layers played a crucial role in preventing overfitting.

3. **Optimization Strategies**: Early stopping and learning rate reduction helped optimize the training process.

4. **Practical Application**: The model was saved in HDF5 format for easy deployment, with a prediction function demonstrating its practical use.

## Conclusion and Future Work

The CNN model developed for American Sign Language (ASL) gesture recognition has demonstrated exceptional performance with a test set accuracy of 96.35%. The methodology, comprising meticulous data acquisition, preprocessing, and optimized convolutional layers, yielded robust results validated by comprehensive evaluation metrics.

Future work could include:
1. Enhancing the dataset with more diverse gestures
2. Exploring advanced CNN architectures
3. Investigating transfer learning methods
4. Refining the model for real-time gesture recognition systems
5. Developing assistive technologies based on the model

Continued refinement will be pivotal in maximizing the model's performance and utility in practical ASL communication contexts, ultimately working towards bridging the communication gap for DHH individuals in various social and professional settings.
