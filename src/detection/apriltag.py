from pyapriltags import Detector
import cv2
import numpy as np
import yaml

# AprilTag Detector Class
class AprilTagDetector:
    def __init__(self, camera_matrix, dist_coeffs, allowed_ids=None):
        self.detector = Detector(families='tag36h11')
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.allowed_ids = allowed_ids

    # Detect AprilTags in the given frame
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if self.allowed_ids is not None:
            detections = [d for d in detections if d.tag_id in self.allowed_ids]

        return detections

    # Estimate pose of detected AprilTags
    def estimate_pose(self, detection, tag_config):
        pose = None
        tags = {}

        for det in detection:
            tag_id = det.tag_id
            center = det.center
            corners = det.corners

            if tag_id == tag_config['hand']['id']:
                tag_size = tag_config['hand']['size']
            elif tag_id == tag_config['piano']['id']:
                tag_size = tag_config['piano']['size']
            else:
                continue

            if self.K is not None and self.dist is not None:
                try:
                    #Corner-Points on apriltag (from center)
                    obj_points = np.array([
                        [-tag_size/2, tag_size/2, 0],
                        [tag_size/2, tag_size/2, 0],
                        [tag_size/2, -tag_size/2, 0],
                        [-tag_size/2, -tag_size/2, 0]
                    ])

                    # Solve PnP to get rotation and translation vectors
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        corners,
                        self.K,
                        self.dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:
                        # Convert rotation vector to rotation matrix
                        R, _ =  cv2.Rodrigues(rvec)
                        pose = {
                            'rotation': R,
                            'translation': tvec.flatten(),
                            'rvec': rvec.flatten(),
                            'tvec': tvec.flatten()
                        }

                except:
                    pass

            # Store tag information
            tags[tag_id] = {
                'id': det.tag_id,
                'center': center,
                'corners': corners,
                'pose': pose
            }

        return tags


# Load AprilTag configuration from YAML file
def load_apriltag_config(path):

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg["apriltags"]
