"""
PianoPoseEstimation - Hand pose estimation for piano playing analysis

Modules:
    - camera: Camera calibration and configuration
    - detection: AprilTag and MediaPipe hand detection
    - geometry: Coordinate transformations
    - visualization: Drawing overlays and debug views
"""

from .camera import load_camera_config
from .detection import AprilTagDetector, load_apriltag_config, MediaPipeHandDetection
from .geometry import CoordinateTransformer
from .visualization import draw_axes, draw_plane
