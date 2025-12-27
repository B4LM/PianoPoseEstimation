# Change this line
from pyapriltags import Detector
import cv2

class AprilTagDetector:
    def __init__(self):
        # Change this line - pyapriltags uses 'Detector' directly
        self.detector = Detector(families="tag36h11") 

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # This remains the same
        return self.detector.detect(gray)