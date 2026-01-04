# Face Mask Detection using Convolutional Neural Networks

A real-time face mask detection system built with CNN that classifies whether a person is wearing a mask correctly, incorrectly, or not at all. Uses OpenCV for video processing and a custom trained model.

## Overview

This project implements a three-class face mask detection system that works in real-time through your webcam. The model distinguishes between properly masked faces, incorrectly masked faces, and unmasked faces.

## Dataset

The model was trained on a face mask detection dataset with XML annotations. The dataset structure includes:

- Training set: 3,850 images
- Test set: 222 images  
- Classes: `with_mask`, `without_mask`, `mask_weared_incorrect`

Images are preprocessed to 128x128 pixels. The dataset should be organized with images in `archive/images/` and corresponding XML annotations in `archive/annotations/`. The preprocessing script automatically splits the data, using the first 800 images for training.

## Requirements

- Python 3.8+
- OpenCV
- Keras/TensorFlow
- NumPy
- PIL/Pillow
- xmltodict
- matplotlib

## Setup

1. Clone the repository
2. Place your dataset in the `archive/` directory with `images/` and `annotations/` subdirectories
3. Run `FaceTraining.ipynb` to train the model (generates `FaceMaskModel.h5`)
4. Make sure `haarcascade_frontalface_alt2.xml` is in the project directory

## Usage

### Training

Run the `FaceTraining.ipynb` notebook to preprocess the data and train the CNN model. The model trains for 35 epochs and saves as `FaceMaskModel.h5`.

### Real-time Detection

Run the detection script:

```bash
python FaceMaskDetection.py
```

Or use the `FaceMaskDetection.ipynb` notebook. The application accesses your default webcam and displays detection results with colored bounding boxes. Press 'x' to exit.

- Green box: Properly masked
- Red box: No mask detected  
- Orange box: Mask worn incorrectly

## Model Architecture

The CNN uses two convolutional blocks with batch normalization, max pooling, and dropout layers. A fully connected layer (512 neurons) feeds into the final softmax layer for 3-class classification. Trained with Adamax optimizer and categorical crossentropy loss.

## Files

- `FaceTraining.ipynb` - Training notebook
- `FaceMaskDetection.py` - Real-time detection script
- `FaceMaskDetection.ipynb` - Detection notebook version
- `haarcascade_frontalface_alt2.xml` - Haar cascade classifier for face detection
- `FaceMaskModel.h5` - Trained model (generated after training)
