# NatyaNetra - AI Based Bharatanatyam Mudra Recognition System

## Overview

NatyaNetra is an AI-powered Bharatanatyam mudra recognition system that combines Computer Vision, Machine Learning, and Embedded Systems.

The system extracts MediaPipe hand landmarks, trains a PyTorch MLP classifier, performs real-time gesture recognition, and communicates with an MSPM0 microcontroller through UART for interactive storytelling.

## Features

- Real-time hand tracking
- MediaPipe landmark extraction
- PyTorch MLP gesture classifier
- UART communication with MSPM0
- Interactive storytelling outputs
- Dataset preparation pipeline
- Model training pipeline

## Technologies

- Python
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- MSPM0G3507
- UART Communication

## Workflow

Dataset Images
↓
MediaPipe Landmark Extraction
↓
Feature Normalization
↓
MLP Training
↓
Real-Time Gesture Recognition
↓
UART Communication
↓
Storytelling Output

## Project Structure

prepare_dataset.py
- Dataset preparation and landmark extraction

train_model.py
- MLP model training

gesture_classifier_ml.py
- Real-time gesture classification

main_ml.py
- Main application with webcam and UART integration

## Results

- Training Accuracy: 99.49%
- Validation Accuracy: 99.72%
- Test Accuracy: 100%

## Future Enhancements

- More mudra classes
- Mobile application support
- Voice narration integration
- Transformer-based gesture recognition
