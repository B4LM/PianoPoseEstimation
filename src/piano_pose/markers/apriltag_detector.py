from pyapriltags import Detector
import cv2
import numpy as np

class AprilTagDetector:
    def __init__(self, camera_matrix, dist_coeffs, allowed_ids=None):
        self.detector = Detector()
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.allowed_ids = allowed_ids

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if self.allowed_ids is not None:
            detections = [d for d in detections if d.tag_id in self.allowed_ids]

        return detections

    def estimate_pose(self, detection, tag_size):
        s = tag_size

        obj_points = np.array([
            [-s/2, -s/2, 0],
            [ s/2, -s/2, 0],
            [ s/2,  s/2, 0],
            [-s/2,  s/2, 0]
        ], dtype=np.float32)

        img_points = detection.corners.astype(np.float32)

        _, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        return rvec, tvec
