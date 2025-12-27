import cv2
import numpy as np

def draw_axes(frame, K, dist, rvec, tvec, length=0.05):
    # 3D-Achsenpunkte
    axes = np.float32([
        [0,0,0],
        [length,0,0],
        [0,length,0],
        [0,0,length]
    ])
    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)

    o = tuple(imgpts[0].ravel())
    x = tuple(imgpts[1].ravel())
    y = tuple(imgpts[2].ravel())
    z = tuple(imgpts[3].ravel())

    cv2.line(frame, o, x, (0,0,255),2)
    cv2.line(frame, o, y, (0,255,0),2)
    cv2.line(frame, o, z, (255,0,0),2)

def draw_plane(
    frame,
    camera_matrix,
    dist_coeffs,
    rvec,
    tvec,
    size_x=0.20,
    size_y=0.06,
    color=(0, 255, 255),
    thickness=2
):
    """
    Zeichnet eine rechteckige Ebene (z = 0) im lokalen AprilTag-Koordinatensystem.

    size_x: Breite (x-Richtung) in Metern
    size_y: Tiefe  (y-Richtung) in Metern
    """

    # 1️⃣ 3D-Punkte der Ebene im Piano-Koordinatensystem
    half_x = size_x / 2.0

    plane_points_3d = np.array([
        [-half_x, 0.0, 0.0],
        [ half_x, 0.0, 0.0],
        [ half_x, size_y, 0.0],
        [-half_x, size_y, 0.0]
    ], dtype=np.float32)

    # 2️⃣ 3D → 2D Projektion
    image_points, _ = cv2.projectPoints(
        plane_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    image_points = image_points.reshape(-1, 2).astype(int)

    # 3️⃣ Linien zeichnen
    for i in range(4):
        pt1 = tuple(image_points[i])
        pt2 = tuple(image_points[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, color, thickness)

    # Optional: Diagonale für Orientierung
    cv2.line(
        frame,
        tuple(image_points[0]),
        tuple(image_points[2]),
        color,
        1
    )

