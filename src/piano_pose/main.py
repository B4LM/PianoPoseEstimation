import cv2
from pathlib import Path

from camera.calibration import load_camera_config
from markers.apriltag_detector import AprilTagDetector
from markers.apriltag_config import load_apriltag_config
from visualisazion.overlay import draw_axes, draw_plane

PROJECT_ROOT = Path(__file__).resolve().parents[2]

camera_cfg = load_camera_config(
    PROJECT_ROOT / "configs" / "camera.yaml"
)

tag_cfg = load_apriltag_config(
    PROJECT_ROOT / "configs" / "apriltags.yaml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["resolution"][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["resolution"][1])
cap.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])

if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden")

detector = AprilTagDetector(
    camera_matrix=camera_cfg["camera_matrix"],
    dist_coeffs=camera_cfg["dist_coeffs"],
    allowed_ids=[tag_cfg.piano.id, tag_cfg.hand.id]
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for det in detections:
        if det.tag_id == tag_cfg.piano.id:
            rvec, tvec = detector.estimate_pose(
                det,
                tag_cfg.piano.size
            )
            # 1️⃣ Koordinatenachsen zeichnen
            draw_axes(
                frame,
                camera_cfg["camera_matrix"],
                camera_cfg["dist_coeffs"],
                rvec,
                tvec,
                length=0.05
            )

            # 2️⃣ Klavierebene zeichnen (x-y-Ebene)
            draw_plane(
                frame,
                camera_cfg["camera_matrix"],
                camera_cfg["dist_coeffs"],
                rvec,
                tvec,
                size_x=0.20,  # Breite der Oktave
                size_y=0.06  # Tiefe der Tasten
            )

        elif det.tag_id == tag_cfg.hand.id:
            rvec, tvec = detector.estimate_pose(
                det,
                tag_cfg.hand.size
            )


    cv2.imshow("Piano AprilTag Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break


'''
#import cv2
#from pathlib import Path
#from camera.calibration import load_camera_config
#from markers.apriltag_detector import AprilTagDetector
from visualisazion.overlay import  draw_axes

# Pfad
PROJECT_ROOT = Path(__file__).resolve().parents[2]
calib = load_camera_config(PROJECT_ROOT / "data/calibration/camera.yaml")

# Kamera
cap = cv2.VideoCapture(0)
detector = AprilTagDetector(tag_size=0.04, camera_matrix=calib["camera_matrix"], dist_coeffs=calib["dist_coeffs"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    for det in detections:
        rvec, tvec = detector.estimate_pose(det)
        draw_axes(frame, calib["camera_matrix"], calib["dist_coeffs"], rvec, tvec)

    cv2.imshow("AprilTag Pose", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''