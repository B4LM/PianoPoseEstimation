import cv2
from markers.apriltag_detector import AprilTagDetector
from pathlib import Path
from camera.calibration import load_camera_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_path = PROJECT_ROOT / "data" / "calibration" / "camera.yaml"

config = load_camera_config(config_path)
print("Using camera:", config["camera_name"])

def draw_tag(frame, detection):
    corners = detection.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    center = tuple(detection.center.astype(int))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    detector = AprilTagDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            draw_tag(frame, det)

        cv2.imshow("AprilTag Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
