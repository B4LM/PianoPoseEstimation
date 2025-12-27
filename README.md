# PianoPoseEstimation
This projects focus is the analysis of  piano-playing. The goal is to use pose estimation of hand movement to detect key-pressing and analyse hand movement.

## Coordinate System

- Origin: AprilTag on piano (left of octave)
- x-axis: along keys (left → right)
- y-axis: depth
- z-axis: upwards (normal to key plane)

Key press is detected when fingertip z < 0
