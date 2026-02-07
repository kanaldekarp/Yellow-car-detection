# Yellow Car Detection and Tracking

This project detects and tracks yellow cars in images and videos using Computer Vision techniques. It utilizes MobileNet SSD for object detection and HSV color filtering for specific color recognition.

## Features

*   **Object Detection**: Uses a pre-trained `MobileNet SSD` (Caffe model) to detect cars in the frame.
*   **Color Filtering**: Applies `HSV` (Hue, Saturation, Value) thresholding to identify *yellow* cars specifically.
*   **Object Tracking**: Implements the `CSRT` tracker from OpenCV to track detected yellow cars across video frames.
*   **Visual Odometry (SLAM)**: Includes a basic monocular visual odometry implementation using ORB features to estimate camera movement (experimental).

## Directory Structure

*   `Codes/`: Contains the Python source scripts.
    *   `Yellow_car_detection_images.py`: Detection on single images.
    *   `Yellow_car_detection_videos.py`: Detection, tracking, and SLAM on videos.
*   `models/`: Contains the pre-trained Caffe models (`deploy.prototxt`, `mobilenet_iter_73000.caffemodel`).
*   `images/`: Sample images for testing.
*   `Videos/`: Sample videos for testing.

## Prerequisites

*   Python 3.x
*   OpenCV (`opencv-python`, `opencv-contrib-python`)
*   NumPy

## Usage

1.  **Image Detection**:
    ```bash
    python Codes/Yellow_car_detection_images.py
    ```

2.  **Video Tracking**:
    ```bash
    python Codes/Yellow_car_detection_videos.py
    ```
