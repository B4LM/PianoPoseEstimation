"""
PianoPoseEstimation - Hand pose estimation for piano playing analysis

Modules:
    - camera: Camera calibration and configuration
    - detection: AprilTag and MediaPipe hand detection
    - geometry: Coordinate transformations
    - visualization: Drawing overlays and debug views
"""

from .camera import CameraCalibrator, CameraManager
from .detection import AprilTagDetector, load_apriltag_config, MediaPipeHandDetection
from .geometry import CoordinateTransformer
from .visualization import draw_axes, draw_plane, draw_fingertip_coords, draw_debug_wrist_to_middle_tip_distance,draw_hand_tag_in_piano_coords, draw_April_tag_box, draw_at_coordinate_system
