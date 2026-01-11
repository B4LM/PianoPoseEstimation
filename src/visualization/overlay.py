import cv2

# Draw fingertip 3D coordinates on the frame
def draw_fingertip_coords(frame, tip_3Dlocations, start_pos=(10, 100), color=(255, 255, 255)):
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

# Draw the bounding box and key points of the AprilTag
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

# Draw calibration status on the frame
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

# Draw the axes of the tag given the axis points
def draw_tag_axes(frame, axis_pts):
    origin = tuple(axis_pts[0])
    x_axis = tuple(axis_pts[1])
    y_axis = tuple(axis_pts[2])
    z_axis = tuple(axis_pts[3])

    cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)   # X - red
    cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)   # Y - green
    cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)   # Z - blue

# Draw key press event on the frame, blue dot at fingertip position
def draw_Key_press_Event(frame, pnt):

    pt = (int(pnt[0]), int(pnt[1]))

    cv2.circle(frame, pt, 30, (255, 0, 0), -1)


# Draw timestamp on the frame
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






