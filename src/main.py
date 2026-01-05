"""
Main entry point for PianoPoseEstimation.
"""
import cv2
import numpy as np
from pathlib import Path

from src.camera import load_camera_config
from src.detection import AprilTagDetector, load_apriltag_config, MediaPipeHandDetection
from src.visualization import draw_axes, draw_plane, draw_vector, draw_line_3d
from src.geometry import CoordinateTransformer


def main():
    """Main pipeline for piano pose estimation."""
    
    # Load configurations
    project_root = Path(__file__).resolve().parents[1]
    
    camera_cfg = load_camera_config(
        project_root / "configs" / "camera.yaml"
    )
    
    tag_cfg = load_apriltag_config(
        project_root / "configs" / "apriltags.yaml"
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["resolution"][1])
    cap.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])
    
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")
    
    # Initialize detectors
    detector = AprilTagDetector(
        camera_matrix=camera_cfg["camera_matrix"],
        dist_coeffs=camera_cfg["dist_coeffs"],
        allowed_ids=[tag_cfg["piano"]["id"], tag_cfg["hand"]["id"]]
    )
    
    hand_detector = MediaPipeHandDetection()
    transformer = CoordinateTransformer(
        camera_cfg["camera_matrix"], 
        camera_cfg.get("dist_coeffs")
    )
    
    print("Starting piano pose estimation... Press ESC to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect AprilTags
        detections = detector.detect(frame)
        
        piano_pose = None
        hand_tag_pose = None
        
        for det in detections:
            if det.tag_id == tag_cfg["piano"]["id"]:
                rvec, tvec = detector.estimate_pose(det, tag_cfg["piano"]["size"])
                piano_pose = (rvec, tvec)
                
                # Draw coordinate axes
                draw_axes(
                    frame,
                    camera_cfg["camera_matrix"],
                    camera_cfg["dist_coeffs"],
                    rvec, tvec,
                    length=0.05
                )
                
                # Draw piano plane
                draw_plane(
                    frame,
                    camera_cfg["camera_matrix"],
                    camera_cfg["dist_coeffs"],
                    rvec, tvec,
                    size_x=0.20,
                    size_y=0.06
                )
                
            elif det.tag_id == tag_cfg["hand"]["id"]:
                rvec, tvec = detector.estimate_pose(det, tag_cfg["hand"]["size"])
                hand_tag_pose = (rvec, tvec)

                # Draw coordinate axes
                draw_axes(
                    frame,
                    camera_cfg["camera_matrix"],
                    camera_cfg["dist_coeffs"],
                    rvec, tvec,
                    length=0.02
                )

                # Draw piano plane
                draw_plane(
                    frame,
                    camera_cfg["camera_matrix"],
                    camera_cfg["dist_coeffs"],
                    rvec, tvec,
                    size_x=0.02,
                    size_y=0.02
                )
        
        # Detect hand landmarks (MediaPipe)
        hand_data = hand_detector.detect(frame)
        
        if hand_data is not None and piano_pose is not None:
            # 1. Estimate Depth
            depth = hand_detector.estimate_hand_depth(hand_data)
            
            # 2. Convert landmarks to 3D Camera Coordinates
            landmarks_3d_camera = transformer.pixel_to_camera_coordinates(
                hand_data['landmarks_normalized'], 
                depth
            )
            
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
                    
            # 6. Visual Feedback
            frame = hand_detector.draw_hand(frame, hand_data, draw_bbox=False)
            
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
        
        cv2.imshow("Piano Pose Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()