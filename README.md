# Automated Egg Counting System

This project implements an Automated Egg Counting System designed to detect and count eggs in real-time on a conveyor belt. The system uses YOLOv5 for object detection to accurately identify eggs as they pass, and a Raspberry Pi for on-device processing, creating a reliable, efficient solution for automated egg counting in production environments.

## Project Overview

This egg counting system reduces human error and streamlines the counting process in egg packaging facilities. By leveraging YOLOv5 for object detection and deploying the model on a Raspberry Pi, the system can identify and log eggs in real-time as they move along a conveyor.

## Key Features

- **Real-Time Counting:** Counts eggs as they pass through the camera’s field of view on the conveyor.
- **Accurate Detection:** YOLOv5’s object detection capabilities enable precise egg recognition.
- **Easy Data Logging:** Displays the count in real-time and logs results for batch analysis.

## Components Needed

- Raspberry Pi 3 or 4
- Camera Module (Pi Camera)
- Lighting Setup (to ensure consistent visibility for the camera)

## Software Requirements
- **Google Colab:** For training the YOLOv5 model on labeled egg images.
- **Python 3:** With dependencies including OpenCV, YOLOv5, and PyTorch.
- **Raspberry Pi OS**

### Python Dependencies

- **PyTorch and TorchVision:** For running YOLOv5 on the Raspberry Pi.
- **OpenCV:** For image capture and processing.
- **Numpy, Matplotlib, and SciPy:** For general data handling and any visualization or calculations.
- **Pandas:** For structured data logging.

## Project Plan

### 1. Set Up Development Environment

- Train YOLOv5 on labeled egg images using Google Colab and save the model in .pt format.
- Set up Raspberry Pi with necessary Python dependencies.

### 2. Configure Hardware

- Connect the camera module to Raspberry Pi and set up lighting to ensure the eggs are clearly visible.

### 3. Software Development

#### 3.1 Model Training with YOLOv5

- Use a labeled dataset of eggs to train YOLOv5 in Google Colab.
- Export the trained model for deployment on Raspberry Pi.

#### 3.2 Real-Time Detection and Counting

- Load the YOLOv5 model on the Raspberry Pi.
- Use OpenCV to capture frames and process each frame through YOLOv5 to detect and count eggs.
- Display the count on-screen or log it for batch record-keeping.
- count only the eggs witch go through the check point .



### 4. Testing and Validation

- Test the system by running known egg batches and validating the logged count accuracy.
- Optimize lighting and camera placement as necessary to improve detection accuracy.

### Final Validation

- **Test Signal:** Validate the system by running a set of test eggs and analyzing the logged count.
- **Optimize:** Fine-tune the model and lighting as needed to improve accuracy.

### how to run 

### Requirements
Python 3.8 or higher.

pip install -r requirements.txt

Download the model (best.pt)

### Install Dependencies:

Use the requirements.txt file to install all necessary libraries:
bash

pip install -r requirements.txt

### Pre-trained Model:

Download the pre-trained model file (best.pth) 
Save the model file in your preferred directory.

## Update the Path to the Model:

Open eggs_counting.py and update the path to point to the location of your model file:
python

model_path = "path/to/your/model/best.pth"  # Update this line
How to Run the Project


bash
pip install -r requirements.txt
Download the model file and update the path in eggs_counting.py.

## Run the project:

bash
python eggs_counting.py

Features
Real-time egg detection using a webcam.
count only the image witch go through the check point 
