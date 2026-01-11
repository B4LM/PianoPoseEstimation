import cv2
import numpy as np

def draw_at_coordinate_system(frame, tag_pose,cam_mtx,dist_coeffs, det, tag_size):

    rvec, tvec = tag_pose
    corners = det.corners.astype(np.float32)
    center = det.center.astype(np.float32)
    axis_length = tag_size
    axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mtx, dist_coeffs)

    origin = tuple(center.ravel().astype(int))
    cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)  # X - rot
    cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)  # Y - grün
    cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)  # Z - blau


def draw_axes(frame, K, dist, rvec, tvec, length=0.05):

    if tvec[2,0] <= 0:
        return  # behind camera

    axes = np.float32([
        [0,0,0],
        [length,0,0],
        [0,length,0],
        [0,0,length]
    ])

    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)

    if not np.isfinite(imgpts).all():
        return

    imgpts = imgpts.reshape(-1,2)
    h, w = frame.shape[:2]

    pts = [tuple(int(v) for v in p) for p in imgpts]

    for p in pts:
        if not (-1000 < p[0] < w+1000 and -1000 < p[1] < h+1000):
            return

    o, x, y, z = pts

    cv2.line(frame, o, x, (0,0,255), 2)
    cv2.line(frame, o, y, (0,255,0), 2)
    cv2.line(frame, o, z, (255,0,0), 2)


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

    # ---------- Guards ----------
    if rvec is None or tvec is None:
        return

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3,1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3,1)

    if tvec[2, 0] <= 0:
        return  # plane behind camera

    # ---------- Geometry ----------
    half_x = size_x / 2.0
    plane_points_3d = np.float32([
        [-half_x, 0.0, 0.0],
        [ half_x, 0.0, 0.0],
        [ half_x, size_y, 0.0],
        [-half_x, size_y, 0.0]
    ])

    # ---------- Projection ----------
    image_points, _ = cv2.projectPoints(
        plane_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    if not np.isfinite(image_points).all():
        return

    image_points = image_points.reshape(-1, 2)

    h, w = frame.shape[:2]
    pts = []

    for p in image_points:
        x, y = int(p[0]), int(p[1])
        if not (-1000 < x < w+1000 and -1000 < y < h+1000):
            return
        pts.append((x, y))

    # ---------- Draw ----------
    for i in range(4):
        cv2.line(frame, pts[i], pts[(i + 1) % 4], color, thickness)

    cv2.line(frame, pts[0], pts[2], color, 1)



def draw_vector(frame, K, dist, rvec, tvec, vector_3d, color=(255, 255, 0)):
    """
    Draw a 3D vector starting from the AprilTag origin.
    
    vector_3d: The end point of the vector in the tag's local coordinate system.
    """
    # Define start (origin) and end points in 3D
    pts_3d = np.array([
        [0, 0, 0],
        vector_3d.flatten()
    ], dtype=np.float32)
    
    # Project to 2D
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    
    # Draw the line
    cv2.line(frame, tuple(pts_2d[0]), tuple(pts_2d[1]), color, 3)
    # Draw a circle at the tip
    cv2.circle(frame, tuple(pts_2d[1]), 5, color, -1)

def draw_line_3d(frame, K, dist, pt1_cam, pt2_cam, color=(255, 255, 0), thickness=2):
    """
    Draw a line between two points in Camera 3D coordinates.
    """
    # pts_cam: already in camera space
    pts_3d = np.array([
        pt1_cam.flatten(),
        pt2_cam.flatten()
    ], dtype=np.float32)
    
    # Project (Identity rotation/translation because points are already in camera space)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    
    cv2.line(frame, tuple(pts_2d[0]), tuple(pts_2d[1]), color, thickness)
    cv2.circle(frame, tuple(pts_2d[1]), 4, color, -1)



def draw_wrist_to_tag_offset(frame, wrist_position, offset,
                                  approx_hand_width_pixels=100, show_info=True):
    """
    visualize wrist to hand april tag offset

    Args:
        frame: image
        wrist: wrist position
        offset: consant offset in meters
        approx_hand_width_pixels: aproximate hand width in pixels
        show_info: bool to hide info
    """
    output = frame.copy()
    wrist = tuple(wrist_position.astype(int))

    if approx_hand_width_pixels > 0:
        pixels_per_meter = approx_hand_width_pixels / 0.085


        offset_x_pixels = int(offset[0] * pixels_per_meter)
        offset_y_pixels = int(offset[1] * pixels_per_meter)

        tag_pos = (wrist[0] + offset_x_pixels, wrist[1] + offset_y_pixels)


        cv2.circle(output, wrist, 10, (255, 0, 255), -1)
        cv2.circle(output, wrist, 10, (0, 0, 0), 2)


        cv2.circle(output, tag_pos, 15, (0, 255, 255), -1)
        cv2.circle(output, tag_pos, 15, (0, 0, 0), 3)

        cv2.arrowedLine(output, wrist, tag_pos,
                        (255, 255, 0), 3, tipLength=0.15)

        # 4. Text
        if show_info:
            cv2.putText(output, "WRIST", (wrist[0] + 15, wrist[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(output, "TAG HERE", (tag_pos[0] + 20, tag_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


            info = f"constant offset: {offset_meters[0] * 100:.1f}cm, {offset_meters[1] * 100:.1f}cm"
            cv2.putText(output, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output

def draw_fingertip_coords(frame, tip_3Dlocations, start_pos=(10, 100), color=(255, 255, 255)):
    '''
    Draw fingertip coordinates-Box-Overlay
    :param frame:
    :param tip_3Dlocations:
    :param labels:
    :param start_pos:
    :param color:
    :return: Overlay
    '''

    x, y = start_pos
    line_height = 25

    #overlay = frame.copy()
    cv2.rectangle(frame, (x - 5, y - 20), (x + 400, y + len(tip_3Dlocations) * line_height), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.5, frame, 0.5, 0, frame)

    fingertips = [
        'THUMB',
        'INDEX',
        'MIDDLE',
        'RING',
        'PINKY']

    for i, pt in enumerate(tip_3Dlocations):
        label = fingertips[i]

        text = f"{label}: X:{round(pt[0]*100, 2)} Y:{round(pt[1]*100, 2)} Z:{round(pt[2]*100, 2)} cm"

        cv2.putText(frame, text, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_debug_wrist_to_middle_tip_distance(frame, distance, start_pos=(10, 100), color=(255, 255, 255)):
    '''
    Draw fingertip coordinates-Box-Overlay
    :param frame:
    :param tip_3Dlocations:
    :param labels:
    :param start_pos:
    :param color:
    :return: Overlay
    '''

    x, y = start_pos
    line_height = 25

    # overlay = frame.copy()
    cv2.rectangle(frame, (x - 5, y - 20), (x + 400, y + line_height), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.5, frame, 0.5, 0, frame)

    #text = f"Distance: {distance} m"
    text = f"Wrist: X:{distance[0]} Y:{distance[1]} Z:{distance[2]} m"

    cv2.putText(frame, text, (x, y + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_hand_tag_in_piano_coords(frame, hand_pose_piano, start_pos=(10, 100), color=(255, 255, 255)):
    '''
    Draw fingertip coordinates-Box-Overlay
    :param frame:
    :param tip_3Dlocations:
    :param labels:
    :param start_pos:
    :param color:
    :return: Overlay
    '''

    x, y = start_pos
    line_height = 30

    # overlay = frame.copy()
    #cv2.rectangle(frame, (x - 5, y - 20), (x + 400, y + line_height), (0, 0, 0), -1)
    #cv2.addWeighted(frame, 0.5, frame, 0.5, 0, frame)

    # text = f"Distance: {distance} m"
    text = f"hand tag: X:{round(hand_pose_piano[0]*100,2)} Y:{round(hand_pose_piano[1]*100,2)} Z:{round(hand_pose_piano[2]*100,2)} cm"

    cv2.putText(frame, text, (x, y + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

def draw_April_tag_box(frame, det):
    (ptA, ptB, ptC, ptD) = det.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    cv2.line(frame, ptA, ptB, (0,255,0), 2)
    cv2.line(frame, ptB, ptC, (0,255,0), 2)
    cv2.line(frame, ptC, ptD, (0,255,0), 2)
    cv2.line(frame, ptD, ptA, (0,255,0), 2)

    cv2.circle(frame, ptA, 10, (255, 0, 255), -1)
    cv2.circle(frame, ptB, 10, (0, 0, 255), -1)
    cv2.circle(frame, ptC, 10, (255, 0, 0), -1)
    cv2.circle(frame, ptD, 10, (0, 255, 255), -1)

    (cX, cY) = (int(det.center[0]), int(det.center[1]))
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

def draw_calibration_status(frame, calibrated: bool):
    h, w = frame.shape[:2]

    if calibrated:
        text = "CALIBRATED"
        color = (0, 255, 0)
    else:
        text = "NOT CALIBRATED"
        color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Top-right position
    x = w - text_width - 10
    y = 50

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def draw_tag_axes(frame, axis_pts):
    origin = tuple(axis_pts[0])
    x_axis = tuple(axis_pts[1])
    y_axis = tuple(axis_pts[2])
    z_axis = tuple(axis_pts[3])

    cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)   # X - red
    cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)   # Y - green
    cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)   # Z - blue

def draw_Key_press_Event(frame, pnt):

    pt = (int(pnt[0]), int(pnt[1]))

    cv2.circle(frame, pt, 30, (255, 0, 0), -1)

import cv2
from datetime import datetime

def draw_timestamp(
    frame,
    session_time,
    *,
    position: tuple[int, int] =(20, 40),
    font_scale: float =3,
    color: tuple[int, int, int] =(0, 255, 0),
    thickness: int =2,
):

    timer = f"Time: {session_time:7.3f} s"

    cv2.putText(
        frame,
        timer,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame






