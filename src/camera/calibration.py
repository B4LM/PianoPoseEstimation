import numpy as np
import cv2
import os

# Camera Calibrator Class
class CameraCalibrator:
    def __init__(self, pattern_size=(8,6),square_size= 0.025):

        self.pattern_size = pattern_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        self.objp *= square_size

        self.objpoints = []  #3D points in real world
        self.imgpoints = []  #2D points in image plane
        self.calibration_images = []

    # Find chessboard corners in images
    def find_chessboard_corners(self, image_paths, showcorners=False):

        self.objpoints = []
        self.imgpoints = []
        self.calibration_images = []

        # Iterate through images and find chessboard corners
        for i, fname in enumerate(image_paths):
            img = cv2.imread(fname)
            if img is None:
                print(f"Could not read {fname}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(self.objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria=self.criteria)

                self.imgpoints.append(corners2)
                self.calibration_images.append(fname)

                # Draw and display the corners
                if showcorners:
                    cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
                    cv2.imshow("Chessboard Corners", img)
                    cv2.waitKey(500)

                print(f"Corners found in image {i+1}/{len(image_paths)}: {fname}")

            else:
                print(f"Could not find chessboard corners in image {i+1}/{len(image_paths)}: {fname}")

        if showcorners:
            cv2.destroyAllWindows()

        return len(self.imgpoints)

    # Calibrate camera using found points
    def calibrate(self, image_size):

        print(f"calibrating started!")

        #OpenCV calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_size, None, None)

        #Calculate projection error, indicator for Exactness of found calibration parameters
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)

            # Compute the error between the detected points and the projected points
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(self.objpoints)

        return mtx, dist, mean_error

    # Save calibration images from camera
    def save_calibration_images(self, camera, num_images, save_dir):

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        print(f"capturing {num_images} calibration images")
        print("press 's' to save image, 'q' to quit early")

        saved_images = []
        count = 0

        # Capture images from camera
        while count < num_images:
            ret, frame = camera.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, f"Captured: {count}/ {num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            cv2.putText(display, "Press 's' to save, or 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),2)
            cv2.imshow("frame", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Check if corners get detected
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, _ = cv2.findChessboardCorners(gray, self.pattern_size, None)
                if ret_corners:
                    filename = f"{save_dir}/calib_{count:03d}.jpg"
                    cv2.imwrite(filename, frame)
                    saved_images.append(filename)
                    count +=1
                    print(f"Image saved: {count} / {num_images}")
                else:
                    print(f"Could not find chessboard corners")

            elif key == ord('q'):
                print("Capturing stopped early")
                break

        cv2.destroyAllWindows()
        return saved_images

# Hand Tag Landmark Calibration Class
class hand_tag_landmark_calibration:
    def __init__(self, number_of_frames: int=30):
        self.number_of_frames = number_of_frames

    # Calibrate hand using collected frames
    def calibrate_hand(self, hand_data, hand_pose,calibration_frames):
        cal_complete = False
        cal_inprogress = True

        t_HT = None
        R_HT = None

        # Collect data for current frame
        frame_data = {
            'tag_translation': np.array(hand_pose['translation']),  # 3-vector
            'tag_rotation': np.array(hand_pose['rotation']),  # 3x3 matrix
            'wrist': np.array([
                hand_data['world_landmarks'][0].x,
                hand_data['world_landmarks'][0].y,
                hand_data['world_landmarks'][0].z
            ]),
            'index': np.array([
                hand_data['world_landmarks'][5].x,
                hand_data['world_landmarks'][5].y,
                hand_data['world_landmarks'][5].z
            ]),
            'pinky': np.array([
                hand_data['world_landmarks'][17].x,
                hand_data['world_landmarks'][17].y,
                hand_data['world_landmarks'][17].z
            ])
        }

        # Append current frame data to calibration frames
        calibration_frames.append(frame_data)

        # Check if enough frames have been collected for calibration
        if len(calibration_frames) >= self.number_of_frames:
            cal_complete = True
            cal_inprogress = False
            print("Calibration complete!")

            # Compute averages
            t_TW_stack = np.stack([f['tag_translation'] for f in calibration_frames])
            wrist_stack = np.stack([f['wrist'] for f in calibration_frames])
            index_stack = np.stack([f['index'] for f in calibration_frames])
            pinky_stack = np.stack([f['pinky'] for f in calibration_frames])

            # Compute average translations and rotations
            t_TW_avg = np.mean(t_TW_stack, axis=0)
            wrist_avg = np.mean(wrist_stack, axis=0)
            index_avg = np.mean(index_stack, axis=0)
            pinky_avg = np.mean(pinky_stack, axis=0)

            # Get rotation from the middle frame
            middle_index = len(calibration_frames) // 2
            R_TW_avg = calibration_frames[middle_index]['tag_rotation']

            print(f"t_TW_avg: {t_TW_avg}")
            print(f"wrist_avg: {wrist_avg}")
            print(f"index_avg: {index_avg}")
            print(f"pinky_avg: {pinky_avg}")
            print(f"R_TW_avg: {R_TW_avg}")

            # Compute hand-to-tag transformation
            X = index_avg - wrist_avg
            X /= np.linalg.norm(X)

            Y = pinky_avg - wrist_avg
            Y /= np.linalg.norm(Y)

            Z = np.cross(X, Y)
            Z /= np.linalg.norm(Z)

            # Re-orthogonalize Y
            R_HW = np.column_stack((X, Y, Z))
            t_HT = R_HW.T @ (t_TW_avg - wrist_avg)
            R_HT = R_TW_avg.T @ R_HW

        return cal_complete, cal_inprogress, t_HT, R_HT




