# PianoPoseEstimation

This project focuses on the 3D pose estimation of a hand using MediaPipe and AprilTags.  

The detection can be started by running the `run.py` file. If any required packages are missing, a message will appear in the terminal with instructions to install them using the provided requirements file.  

If a new camera is used, it must be documented in the YAML file as a new entry. Since a new camera is not calibrated by default, it can be calibrated at startup using a chessboard pattern.  

Fingertip position estimation will begin after the hand-mounted AprilTag has been calibrated. To perform the calibration, hold the hand flat and still within the camera view (both the hand and the tag must be detected) and press `s`. After a few seconds (30 frames), the calibration will be completed.  

If other AprilTags or sizes are used, the corresponding configuration in the YAML file must be adapted.  

## Global Coordinate System (stationary AprilTag)

- **Origin:** Center of the stationary AprilTag  
- **x-axis:** Left → Right (from the perspective of the user)  
- **y-axis:** Depth → Away from the user  
- **z-axis:** Upwards (normal to the AprilTag)