# Basketball Shot - Performance Analysis

**Author:** Dusan Vicentic  
**Course:** CS 898BA - Image Analysis & Computer Vision 
**Phase:** Midterm Progress & Baseline Implementation  

## Project Overview
This project aims to mathematically define an individual player's "perfect shot arc" and predict mechanical deviations using dense optical flow. For the midterm report, the system relies on domain engineering and kinematics to extract a mathematical trajectory model from a static-angle video, bypassingHuman Pose Estimation (HPE) neural networks.

## Setup

### Dependencies
Ensure you have Python installed along with the following libraries:
*   `opencv-python` (`cv2`)
*   `numpy`
*   `matplotlib`

## Execution Steps: 
1. Place a basketball shot video in root directory (it should be trimmed to show release to entry)
2. Update 'video_path' to match the video file name
3. Run script 'main.py'

## Explenation: 
The current script utilizies 3 computer vision techniques:
1. Gaussiam Mixture background subtraction
2. Hough Circle transform:
3. Polynomial Regression

## Results
A plot output will be created under 'ball_trajectory_baseline.jpg'
The midterm implementation successfully detects the trajectory of a basketball and plots the arc of that shot