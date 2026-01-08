
import cv2
import numpy as np
from pathlib import Path

from src.camera import CameraCalibrator
from src.camera import CameraManager
from src.detection import AprilTagDetector, load_apriltag_config, MediaPipeHandDetection
from src.visualization import draw_axes, draw_plane, draw_vector, draw_line_3d
from src.geometry import CoordinateTransformer
from src.visualization.overlay import draw_fingertip_coords, draw_debug_wrist_to_middle_tip_distance,draw_hand_tag_in_piano_coords, draw_April_tag_box, draw_at_coordinate_system


def main():
    """Main pipeline for piano pose estimation."""
    
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
    
    # Initialize detectors
    yaml_camera_mtx = camera_cfg["camera_matrix"]
    camera_mtx = np.array(yaml_camera_mtx, dtype=np.float64)
    yaml_camera_dist = camera_cfg["dist_coeffs"]
    camera_dist = np.array(yaml_camera_dist, dtype=np.float64)

    april_detector = AprilTagDetector(
        camera_matrix=camera_mtx,
        dist_coeffs=camera_dist,
        allowed_ids=[tag_cfg["piano"]["id"], tag_cfg["hand"]["id"]]
    )
    
    hand_detector = MediaPipeHandDetection()
    transformer = CoordinateTransformer(
        camera_matrix = camera_mtx,
        dist_coeffs = camera_dist
    )
    
    print("Starting piano pose estimation... Press ESC to quit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Detect AprilTags
        april_detections = april_detector.detect(frame)
        
        piano_tag_pose = None
        hand_tag_pose = None
        
        for det in april_detections:
            if det.tag_id == tag_cfg["piano"]["id"]:
                rvec, tvec = april_detector.estimate_pose(det, tag_cfg["piano"]["size"])
                #print("tvec:", tvec.ravel())
                R, _ = cv2.Rodrigues(rvec)
                print("X·Z =", np.dot(R[:, 0], R[:, 2]))
                print("Y·Z =", np.dot(R[:, 1], R[:, 2]))
                print("X×Y =", np.cross(R[:, 0], R[:, 1]))
                print("Z   =", R[:, 2])
                #z_axis = R[:, 2]
                #print("Tag-Normale:", z_axis)
                draw_April_tag_box(frame, det)
                piano_tag_pose = (rvec, tvec)
                piano_transform_mtx = transformer.get_transformation_matrix(*piano_tag_pose)

                # Draw coordinate axes
                if rvec is not None and tvec is not None:
                    draw_at_coordinate_system(frame, piano_tag_pose, camera_mtx,camera_dist, det, tag_cfg["piano"]["size"])
                    '''
                    draw_axes(
                        frame,
                        camera_mtx,
                        camera_dist,
                        rvec, tvec,
                        length=0.05
                    )

                
                # Draw piano plane
                draw_plane(
                    frame,
                    camera_mtx,
                    camera_dist,
                    rvec, tvec,
                    size_x=0.20,
                    size_y=0.06
                )
                '''

            elif det.tag_id == tag_cfg["hand"]["id"]:
                rvec, tvec = april_detector.estimate_pose(det, tag_cfg["hand"]["size"])
                hand_tag_pose = (rvec, tvec)
                hand_transform_mtx = transformer.get_transformation_matrix(*hand_tag_pose)

                # Draw coordinate axes
                if rvec is not None and tvec is not None:
                    draw_at_coordinate_system(frame, hand_tag_pose, camera_mtx,camera_dist, det, tag_cfg["hand"]["size"])
                '''
                    draw_axes(
                        frame,
                        camera_mtx,
                        camera_dist,
                        rvec, tvec,
                        length=0.02
                    )

                
                # Draw piano plane
                draw_plane(
                    frame,
                    camera_mtx,
                    camera_dist,
                    rvec, tvec,
                    size_x=0.02,
                    size_y=0.02
                )
                '''

        
        # Detect hand landmarks (MediaPipe)
        hand_data = hand_detector.detect(frame)
        
        if hand_data is not None and piano_tag_pose is not None:
            '''
            # 1. Estimate Depth
            depth = hand_detector.estimate_hand_depth(hand_data)
            
            # 2. Convert landmarks to 3D Camera Coordinates
            landmarks_3d_camera = transformer.pixel_to_camera_coordinates(
                hand_data['landmarks_normalized'], 
                depth
            )
            '''

            if hand_tag_pose is not None:
                image_size = np.array([camera_cfg["image_width"], camera_cfg["image_height"]])
                #landmarks_3d_camera = transformer.get_landmark_3d_coords(hand_data['landmarks_pixel'], hand_tag_pose, image_size)
                #landmarks_3d_camera = transformer.get_landmark_3d_coords(hand_data['mp_landmarks'], image_size, hand_tag_pose)
                #landmarks_3d_piano = transformer.camera_to_piano_coordinates(
                    #landmarks_3d_camera,
                    #piano_tag_pose)
                #hand_tag_to_wrist_vec = np.array([0, 0.03, -0.015])
                #landmark_3D_piano_coords = transformer.world_landmarks_to_piano_transformation(hand_data['world_landmarks'],hand_tag_to_wrist_vec, hand_transform_mtx, piano_transform_mtx)


                ##### Test
                #direct_3d_coords = hand_data['world_landmarks']
                #coords_direct = []
                #for w_lm in direct_3d_coords:
                    #coords_direct.append([w_lm.x, w_lm.y, w_lm.z])
                #coords_direct_array = np.array(coords_direct)
                #fingertip_coords = coords_direct_array[[4, 8, 12, 16, 20]]
                hand_tag_origin = np.array([[0.0, 0.0, 0.0]])
                hand_tag_piano = transformer.hand_to_piano_transform(hand_tag_origin, hand_transform_mtx,
                                                                     piano_transform_mtx)
                draw_hand_tag_in_piano_coords(frame, hand_tag_piano)

                #######



                #fingertip_coords = landmark_3D_piano_coords[[4, 8, 12, 16, 20]]
                #draw_fingertip_coords(frame, fingertip_coords)
                #wrist2middletip = transformer.get_hand_size_meters(hand_data['world_landmarks'])
                #draw_debug_wrist_to_middle_tip_distance(frame,wrist2middletip)
            else:
                print("Warning: No hand tag detected, skipping 3D transformation")
                landmarks_3d_camera = None



            '''
            # 3. Handle Wrist-to-Tag Offset Debug Vector
            if hand_tag_pose is not None:
                wrist_cam = landmarks_3d_camera[0] # Wrist index 0
                tag_rvec, tag_tvec = hand_tag_pose
                tag_cam = tag_tvec.flatten()
                
                # Draw vector from Wrist to Tag
                draw_line_3d(frame, camera_cfg["camera_matrix"], camera_cfg["dist_coeffs"], 
                             wrist_cam, tag_cam, color=(0, 255, 255))
                
                # Show distance in text
                dist_offset = np.linalg.norm(wrist_cam - tag_cam)
                cv2.putText(frame, f"Tag-Wrist Offset: {dist_offset:.3f}m", (50, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 4. Transform Landmarks to Piano Coordinates
            landmarks_3d_piano = transformer.camera_to_piano_coordinates(
                landmarks_3d_camera, 
                piano_pose
            )
            
            
            # 5. Check for Key Presses (z < 0)
            pressing_fingers = []
            for finger_name, tip_info in hand_data['finger_tips'].items():
                idx = tip_info['index']
                point_piano = landmarks_3d_piano[idx]
                
                if transformer.is_point_pressing_key(point_piano, threshold=0.005): # 5mm threshold
                    pressing_fingers.append(finger_name)
            '''
            # 6. Visual Feedback
            frame = hand_detector.draw_hand(frame, hand_data, draw_bbox=False)
            '''
            if pressing_fingers:
                text = f"KEY PRESSED: {', '.join(pressing_fingers)}"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            # Debug: show wrist Z
            wrist_piano_z = landmarks_3d_piano[0][2]
            cv2.putText(frame, f"Wrist Z: {wrist_piano_z:.3f}m", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif hand_data is not None:
            # Fallback if no piano tag: just draw hand
            #offset = np.array([0.8, -0.6])  # Your values
            frame = hand_detector.draw_hand(frame, hand_data, draw_bbox= False)
            wrist_pos = hand_data['wrist_position']
            #OFFSET = np.array([0.04, 0.0, 0.01])
            #frame = draw_wrist_to_tag(frame, wrist_pos, offset= OFFSET)
        '''
        scale = 0.5  # 50% of the original size
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Piano Pose Estimation", frame_small)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()