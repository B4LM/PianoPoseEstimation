
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
import time

from src.camera import CameraCalibrator, hand_tag_landmark_calibration
from src.camera import CameraManager
from src.detection import AprilTagDetector, load_apriltag_config, MediaPipeHandDetection
from src.geometry import CoordinateTransformer
from src.visualization import draw_fingertip_coords,draw_April_tag_box, draw_calibration_status, draw_tag_axes, draw_Key_press_Event, draw_timestamp


def main():
    
    # Load configurations
    project_root = Path(__file__).resolve().parents[1]

    camera_manager = CameraManager(project_root / "configs" / "camera.yaml")

    camera_name = camera_manager.current_camera
    camera_cfg = camera_manager.cameras[camera_name]

    print(f"Current camera: {camera_name}")
    print(f"Calibrated: {camera_cfg["calibration"]["calibrated"]}")

    ###### camera calibration #######

    if not camera_cfg["calibration"]["calibrated"]:
        print("\n camera needs calibration!")

        response = input("Calibrate now? (y/n): ").lower()

        if response == "y":
            # calibration
            # initiate calibration
            calibrator = CameraCalibrator(pattern_size = (8,6), square_size = 0.025)

            #open camera
            cap = cv2.VideoCapture(camera_cfg.get("device_id"),0)

            print("\n ------Calibration Mode------")
            save_dir = f"calibration_images_{camera_name}"
            image_paths = calibrator.save_calibration_images(
                cap, num_images= 15, save_dir = save_dir
            )

            if len(image_paths) >= 5:
                points_found = calibrator.find_chessboard_corners(image_paths, showcorners=True)

                if points_found >= 5:
                    first_img = cv2.imread(image_paths[0])
                    image_size = first_img.shape[:2]
                    mtx, dist, mean_error = calibrator.calibrate(image_size)

                    if mean_error is not None and mean_error < 0.5:
                        camera_manager.update_calibration(camera_name, mtx, dist, mean_error)
                        print(f"calibration complete!")
                        print(f"arithmetical mean of the errors: {mean_error:.4f}")
                        print(f"camera matrix: \n {mtx}")
                        print(f"distortion coefficients: {dist}")

                    else:
                        print(f"calibration not successful, mean error to high: {mean_error:.4f}")
                else:
                    print(f"calibration not successful, not enough chessboard-points detected")
            else:
                print(f"calibration not successful, not enough calibration images")

            cap.release()
            cv2.destroyAllWindows()

        elif response == "n":
            print("proceed with uncalibrated camera")

    ###### End of calibration #######
    
    tag_cfg = load_apriltag_config(
        project_root / "configs" / "apriltags.yaml"
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_cfg.get("device_id"), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["image_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["image_height"])
    cap.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")

    # save frames as video
    recordings_dir = project_root / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = recordings_dir / f"piano_pose_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = camera_cfg["fps"]
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        raise RuntimeError("VideoWriter could not be opened")

    #save fingertip pos-logs
    log_path = recordings_dir / f"finger_coords_{timestamp}.csv"

    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["time", "frame", "finger", "x", "y", "z"])

    start_time = time.time()
    frame_idx = 0

    # get camera intrinsics
    yaml_camera_mtx = camera_cfg["camera_matrix"]
    camera_mtx = np.array(yaml_camera_mtx, dtype=np.float64)
    yaml_camera_dist = camera_cfg["dist_coeffs"]
    camera_dist = np.array(yaml_camera_dist, dtype=np.float64)

    # Initialize detectors
    april_detector = AprilTagDetector(
        camera_matrix=camera_mtx,
        dist_coeffs=camera_dist,
        allowed_ids=[tag_cfg["piano"]["id"], tag_cfg["hand"]["id"]]
    )
    
    hand_detector = MediaPipeHandDetection()
    hand_calibrator = hand_tag_landmark_calibration()
    transformer = CoordinateTransformer(
        camera_matrix = camera_mtx,
        dist_coeffs = camera_dist
    )

    piano_tag_id = tag_cfg["piano"]["id"]
    hand_tag_id = tag_cfg["hand"]["id"]
    cal_complete = False
    cal_inprogress = False
    calibration_frames = []

    # main loop
    print("Starting piano pose estimation... Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        current_time = time.time() - start_time
        if not ret:
            break

        # Detect AprilTags
        april_detections = april_detector.detect(frame)
        detected_tags = april_detector.estimate_pose(april_detections,tag_cfg)

        #looping through apriltag detections
        for det in april_detections:
            if det.tag_id == piano_tag_id:
                piano_pose = detected_tags[piano_tag_id]["pose"]

                #draw bounding box and axis
                draw_April_tag_box(frame, det)
                piano_tag_axes = transformer.get_apriltag_axes(piano_pose)
                draw_tag_axes(frame, piano_tag_axes)

            elif det.tag_id == tag_cfg["hand"]["id"]:
                hand_pose = detected_tags[hand_tag_id]["pose"]

                # draw bounding box and axis
                draw_April_tag_box(frame, det)
                hand_tag_axes = transformer.get_apriltag_axes(hand_pose)
                draw_tag_axes(frame, hand_tag_axes)

        # detect hand landmarks (MediaPipe)
        hand_data = hand_detector.detect(frame)

        if hand_data is not None:

            # coordinate transformations
            if piano_tag_id in detected_tags and hand_tag_id in detected_tags:

                # pose of hand-apriltag to piano-apriltag (origin)
                hand_pose_piano = transformer.hand_to_piano_transform(detected_tags, piano_tag_id, hand_tag_id)

                # calibration of hand-pose-estimated landmark-positions to hand-apriltag
                key = cv2.waitKey(1) & 0xFF

                # hold hand flat and still, then start calibration by pressing "s" key
                if (key == ord('s') or cal_inprogress) and not cal_complete:

                    # calibration gets transform from wrist-knuckle plane (almost stiff) induced coordinate-system to hand-apriltag
                    cal_complete, cal_inprogress, t_lm_to_hand, R_lm_to_hand =  hand_calibrator.calibrate_hand(hand_data, hand_pose_piano, calibration_frames)

                if cal_complete:
                    try:
                        fingertip_indices = [4, 8, 12, 16, 20]
                        fingertip_coords = [hand_data['world_landmarks'][i] for i in fingertip_indices]

                        # transform world-landmarks positions to piano-coordinates (through hand-apriltag)
                        fingertip_coords_piano = [transformer.worldlandmark_to_piano_transform(t_lm_to_hand, R_lm_to_hand,i,hand_pose) for i in fingertip_coords]

                        # log fingertip positions
                        for finger_id, pt in zip(fingertip_indices, fingertip_coords_piano):
                            csv_writer.writerow([
                                current_time,
                                frame_idx,
                                finger_id,
                                float(pt[0]),
                                float(pt[1]),
                                float(pt[2])
                            ])

                        # keypress event -> Fingertip z-pos < 0
                        for i, pt in enumerate(fingertip_coords_piano):
                            if pt[2] < 0:
                                tip_pos = hand_data['landmarks_pixel'][fingertip_indices[i]]
                                draw_Key_press_Event(frame, tip_pos)

                    except:
                        print("no finger coords!")
                        fingertip_coords_piano = None

                    if fingertip_coords_piano is not None:
                        draw_fingertip_coords(frame,fingertip_coords_piano)

            draw_calibration_status(frame, cal_complete)
            frame = hand_detector.draw_hand(frame, hand_data, draw_bbox=False)

        # for logging
        frame_idx += 1

        # get timer position
        timer_x = int(camera_cfg['image_width'] / 2)
        timer_y = int(camera_cfg['image_height'] - 10)
        draw_timestamp(frame, current_time, position=(timer_x, timer_y))

        # save frame in video
        video_writer.write(frame)

        # scale for window, if to big
        scale = 0.5
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Piano Pose Estimation", frame_small)

        # stop program
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()