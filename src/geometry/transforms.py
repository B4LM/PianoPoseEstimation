import numpy as np
import cv2
import math
from typing import Optional, Dict, Tuple

class CoordinateTransformer:
    '''
    Class to handle coordinate transformations between AprilTag and camera coordinate systems.
    '''
    def __init__(self, camera_matrix, dist_coeffs):
        '''
        initialize the transformer with camera parameters
        :param camera_matrix: 3x3 intrinsic camera matrix
        :param dist_coeffs: distortion coefficients
        '''
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))

    def april_tag_to_camera_pose(self, tag_corners, tag_size):
        '''
        Estimate the pose of the AprilTag relative to the camera.
        :param tag_corners: 4 corners of the detected piano-AprilTag in image coordinates
        :param tag_size: physical size of the piano-AprilTag (edge length in meters)
        :return:
        '''

        half_size = tag_size / 2.0

        # 3D object points of the AprilTag corners in the tag coordinate system
        # Origin of global coordinate system at center of the tag
        obj_points = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size, half_size, 0]
        ], dtype=np.float32)

        img_points = tag_corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        return rvec, tvec, success

    def get_transformation_matrix(self, rvec, tvec):
        '''
        Convert rotation and translation vectors to a 4x4 transformation matrix.
        :param rvec: rotation vector (3x1
        :param tvec: translation vector (3x1)
        :return: homogeneous transformation matrix (4x4)
        '''

        #convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        #build homogeneous transformation matrix
        T= np.eye(4)
        T[0:3, 0:3] = R # set rotation part (3x3), top left
        T[0:3, 3] = tvec.flatten()# set translation part (3x1), top right

        return T

    def transform_points(self, points_3d, transformation_matrix):
        '''
        Transform 3D points using the given transformation matrix.
        :param points_3d: Nx3 array of 3D points
        :param transformation_matrix: 4x4 homogeneous transformation matrix
        :return: Nx3 array of transformed 3D points
        '''

        # Convert points to homogeneous coordinates
        points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])

        # Apply transformation
        transformed_points_homogeneous = (transformation_matrix @ points_homogeneous.T).T

        # Convert back to Cartesian coordinates
        transformed_points = transformed_points_homogeneous[:, 0:3] / transformed_points_homogeneous[:, 3:]

        return transformed_points

    def camera_to_piano_coordinates(self, points_camera, piano_transform_mtx):
        '''
        Transform points from camera coordinate system to piano coordinate system.
        :param points_camera: Nx3 array of 3D points in camera coordinates
        :param piano_tag_pose: (rvec, tvec) of piano tag
        :return: Nx3 array of 3D points in piano coordinates

        piano coordinates are used as global coordinates!
        '''

        # Get transformation matrix from camera to piano tag
        #T_camera_to_piano = self.get_transformation_matrix(*piano_tag_pose)

        # Invert to get transformation from piano tag to camera
        T_piano_to_camera = np.linalg.inv(piano_transform_mtx)

        # Transform points
        points_piano = self.transform_points(points_camera, T_piano_to_camera)

        return points_piano

    def camera_to_hand_coordinates(self, points_camera, hand_transform_mtx):
        '''
        Transform points from camera coordinate system to piano coordinate system.
        :param points_camera: Nx3 array of 3D points in camera coordinates
        :param piano_tag_pose: (rvec, tvec) of piano tag
        :return: Nx3 array of 3D points in piano coordinates

        piano coordinates are used as global coordinates!
        '''

        # Get transformation matrix from camera to piano tag
        #T_camera_to_piano = self.get_transformation_matrix(*piano_tag_pose)

        # Invert to get transformation from piano tag to camera
        T_hand_to_camera = np.linalg.inv(hand_transform_mtx)

        # Transform points
        points_hand = self.transform_points(points_camera, T_hand_to_camera)

        return points_hand

    def hand_to_camera_coordinates(self, points_camera, hand_transform_mtx):
        '''
        Transform points from camera coordinate system to piano coordinate system.
        :param points_camera: Nx3 array of 3D points in camera coordinates
        :param piano_tag_pose: (rvec, tvec) of piano tag
        :return: Nx3 array of 3D points in piano coordinates

        piano coordinates are used as global coordinates!
        '''

        # Transform points
        points_hand = self.transform_points(points_camera, hand_transform_mtx)

        return points_hand

    def hand_landmarks_to_piano_coordinates(self,
                                            mediapipe_landmarks_2d: np.ndarray,
                                            hand_tag_pose: tuple,
                                            piano_tag_pose: tuple,
                                            hand_tag_size: float,
                                            depth_scale_factor: float) -> np.ndarray:
        '''
        Convert MediaPipe 2D hand landmarks to 3D piano coordinates.
        :param hand_landmarks: MediaPipe hand landmarks in 2D image coordinates
        :param piano_tag_pose: (rvec, tvec) of piano tag
        :param hand_depth_estimate: estimated depth of the hand in meters
        :param hand_tag_pose: optional (rvec, tvec) of hand tag
        :return: 21x3 array of 3D hand landmarks in piano coordinates
        '''

        if not hand_tag_pose or not piano_tag_pose:
            return None

        T_camera_to_hand_tag = self.get_transformation_matrix(*hand_tag_pose)
        T_camera_to_piano_tag = self.get_transformation_matrix(*piano_tag_pose)

        T_hand_tag_to_piano = np.linalg.inv(T_camera_to_piano_tag) @ T_camera_to_hand_tag

        HAND_TAG_OFFSET = np.array([0.04, 0.0, 0.01])

        T_offset = np.eye(4)
        T_offset[0:3, 3] = HAND_TAG_OFFSET
        T_total = T_hand_tag_to_piano @ T_offset

    #tag2wrist_offset = np.array([0, 0, 0])


        landmarks_3d_piano = []
        for landmark in mediapipe_landmarks_2d:
            landmark_3d_piano = self.estimate_landmark_3d_from_2d_relative(landmark, hand_tag_size, depth_scale_factor)

            #landmark_3d_piano = landmark_3d_hand #+ tag2wrist_offset

            point_homogeneous = np.append(landmark_3d_piano, 1)

            point_piano_homogeneous = T_total @ point_homogeneous
            point_piano = point_piano_homogeneous[:3] / point_piano_homogeneous[3]

            landmarks_3d_piano.append(point_piano)

        return np.array(landmarks_3d_piano)



    def distance_to_piano_surface(self,point_piano):
        '''
        Calculate the distance of a point in piano coordinates to the piano surface (z=0).
        :param point_piano: 3D point in piano coordinates
        :return: distance to piano surface in meters
        '''
        return point_piano[2]  # z-coordinate represents height above piano surface

    def is_point_pressing_key(self, point_piano, threshold=0.01):
        '''
        Determine if a point in piano coordinates is pressing a key.
        :param point_piano: 3D point in piano coordinates
        :param threshold: distance threshold to consider as pressing a key (in meters)
        :return: True if pressing a key, False otherwise
        '''
        distance = self.distance_to_piano_surface(point_piano)
        return distance < threshold

    '''
    def estimate_landmark_3d_from_2d_relative(self,
                                              landmark_2d: np.ndarray,
                                              hand_tag_size: float,
                                              depth_scale_factor: float = 1.0) -> np.ndarray:

        x_rel = landmark_2d[0] * hand_tag_size
        y_rel = landmark_2d[1] * hand_tag_size
        z_rel = landmark_2d[2] * hand_tag_size * depth_scale_factor

        return np.array([x_rel, y_rel, z_rel])
    '''
    
    def pixel_to_camera_coordinates(self, landmarks_normalized: np.ndarray, depth: float) -> np.ndarray:
        '''
        Convert normalized MediaPipe landmarks (0-1) to 3D camera coordinates.
        :param landmarks_normalized: Nx3 array of normalized landmarks
        :param depth: Estimated depth of the hand centers (meters)
        :return: Nx3 array of 3D points in camera coordinate system
        '''
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        landmarks_3d_camera = []
        for lm in landmarks_normalized:
            # MediaPipe normalized x, y are [0, 1]
            # We assume the depth 'depth' is the distance to the wrist or palm center
            # lm[2] is the relative depth from MediaPipe. 
            # We scale it to meters (approx hand depth 0.1m)
            z_cam = depth + lm[2] * 0.1 
            
            x_cam = (lm[0] - 0.5) * (2.0 * cx / fx) * z_cam # Roughly mapping
            # Better unprojection:
            # x_pixel = lm[0] * width
            # x_cam = (x_pixel - cx) * z_cam / fx
            # Since lm[0] is normalized, we need the resolution if we want it exact,
            # but cx/fx is basically half-width-in-meters at z=1.
            
            # Using standard pinhole camera model assuming lm[0/1] are 0-1 mapped to full sensor
            # Let's assume width/height are roughly 2*cx and 2*cy
            x_pixel = lm[0] * (2 * cx)
            y_pixel = lm[1] * (2 * cy)
            
            x_cam = (x_pixel - cx) * z_cam / fx
            y_cam = (y_pixel - cy) * z_cam / fy
            
            landmarks_3d_camera.append([x_cam, y_cam, z_cam])

        return np.array(landmarks_3d_camera)

    def get_tag_to_wrist_vector(self, wrist_camera_3d: np.ndarray, tag_tvec: np.ndarray) -> np.ndarray:
        '''
        Calculate the 3D vector from the Tag origin to the Wrist position in Tag space.
        :param wrist_camera_3d: 3D position of wrist in camera space (3x1 or 1x3)
        :param tag_tvec: tag translation vector from solvePnP (3x1)
        :return: 3D vector from tag to wrist
        '''
        return wrist_camera_3d.flatten() - tag_tvec.flatten()

    def camera_to_tag_space(self, points_camera: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        '''
        Transform 3D points from camera coordinates to a local tag's coordinate system.
        '''
        T_cam_to_tag = self.get_transformation_matrix(rvec, tvec)
        T_tag_to_cam = np.linalg.inv(T_cam_to_tag)
        
        return self.transform_points(points_camera, T_tag_to_cam)

    def piano_to_camera_pose(self, piano_tag_pose):
        ''' Returns the transformation matrix from piano to camera. '''
        return self.get_transformation_matrix(*piano_tag_pose)

    def get_landmark_3d_coords(self,mp_landmarks, image_size: np.ndarray, hand_tag_pose,hand_width: float = 0.1) -> np.ndarray:
        """
        add depth to landmark camera-coordinates
        :param landmarks_pixel: result from hand detection-> position of landmarks
        :param hand_tag_pose: result from hand_april_tag-detection-> position of hand tag
        :param hand_width: hand width in meters
        :param image_size: width and height in pixels
        :return: landmark-positions in 3D camera coordinates
        """
        if hand_tag_pose is None:
            print("Warning: hand_tag_pose is None, cannot compute 3D coordinates")

        hand_tag_center = np.array([0,0,0,1]) #homogenous coordinates
        wrist_hand_tag_vec = np.array([0.02, -0.01, 0, 0]) #homogenous coordinates
        wrist_relocation = hand_tag_center + wrist_hand_tag_vec

        wrist_position_camera = self.get_transformation_matrix(*hand_tag_pose) @ wrist_relocation
        wrist_distance_to_camera = np.linalg.norm(wrist_position_camera[:3])

        cam_LM_coords = []
        for landmark in mp_landmarks.landmark:
            x_cam = int(landmark.x * image_size[0])
            y_cam = int(landmark.y * image_size[1])
            z_cam = wrist_distance_to_camera - hand_width * landmark.z
            cam_LM_coords.append((x_cam, y_cam, z_cam))

        return np.array(cam_LM_coords)

    def shift_world_LM_origin_to_wrist_camera(self,world_landmarks):

        wrist_pos = world_landmarks[0]
        shifted_landmarks = []

        for w_lm in world_landmarks:
            lm_x = w_lm.x - wrist_pos.x
            lm_y = w_lm.y - wrist_pos.y
            lm_z = w_lm.z - wrist_pos.z
            shifted_landmarks.append([lm_x, lm_y, lm_z])

        return np.array(shifted_landmarks)

    def world_landmarks_to_piano_transformation(self,world_landmarks, hand_tag_to_wrist_vec, hand_transform_mtx, piano_transform_mtx):

        #shift landmarks coordinates to wrist (camera coords)
        shifted_wrist_landmarks_camera = self.shift_world_LM_origin_to_wrist_camera(world_landmarks)

        #transform shifted landmarks to hand_tag
        shifted_wrist_landmarks_hand_tag = self.camera_to_hand_coordinates(shifted_wrist_landmarks_camera, hand_transform_mtx)

        #add wrist to hand_tag distance
        shifted_hand_tag_landmarks_hand_tag=shifted_wrist_landmarks_hand_tag + hand_tag_to_wrist_vec

        #transform back to camera
        shifted_hand_tag_landmarks_camera = self.hand_to_camera_coordinates(shifted_hand_tag_landmarks_hand_tag, hand_transform_mtx)

        #transform landmarks to piano coords
        landmarks_piano = self.camera_to_piano_coordinates(shifted_hand_tag_landmarks_camera, piano_transform_mtx)

        return landmarks_piano

    def get_hand_size_meters(self, world_landmarks):
        wrist = world_landmarks[0]
        middle_tip = world_landmarks[12]

        # Euclidean distance in 3D
        dist = math.sqrt(
            (middle_tip.x - wrist.x) ** 2 +
            (middle_tip.y - wrist.y) ** 2 +
            (middle_tip.z - wrist.z) ** 2
        )
        return wrist  # Should be roughly 0.1 to 0.18 meters

    def hand_to_piano_transform(self,
                                tags: Dict[int, Dict],
                                piano_tag_id: int,
                                hand_tag_id: int) -> Optional[Dict]:

        if piano_tag_id not in tags or hand_tag_id not in tags:
            return None

        piano_tag = tags[piano_tag_id]
        hand_tag = tags[hand_tag_id]

        relative_3d = None
        if piano_tag['pose'] is not None and hand_tag['pose'] is not None:
            R_piano = piano_tag['pose']['rotation']
            t_piano = piano_tag['pose']['translation']
            R_hand = hand_tag['pose']['rotation']
            t_hand = hand_tag['pose']['translation']

            R_piano_inv = R_piano.T #since rotational matrix is orthogonal

            relative_translation = R_piano_inv @ (t_hand - t_piano)
            relative_rotation = R_piano_inv @ R_hand

            relative_3d = {
                'translation': relative_translation,
                'rotation': relative_rotation,
                'distance': np.linalg.norm(relative_translation)
            }

        return relative_3d



        points_camera = self.hand_to_camera_coordinates(points,hand_transform_mtx)

        hand_piano = self.camera_to_piano_coordinates(points_camera, piano_transform_mtx)

        return hand_piano

    def worldlandmark_to_piano_transform(
            self,
            t_lm_to_hand,
            R_lm_to_hand,
            LM_pos,
            hand_pose
    ):
        # Convert MediaPipe Landmark → numpy vector
        p_L_H = np.array(
            [LM_pos.x, LM_pos.y, LM_pos.z],
            dtype=np.float32
        )

        R_hand_to_piano = np.array(hand_pose['rotation'])
        t_hand_to_piano = np.array(hand_pose['translation'])

        lm_pos_piano = (
                R_hand_to_piano @ (R_lm_to_hand @ p_L_H + t_lm_to_hand)
                + t_hand_to_piano
        )

        return lm_pos_piano

    def get_apriltag_axes(self, tag_pose, cam_mtx, axis_length:  float= 0.05):

        axes_3d = np.array([
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # +X
            [0, axis_length, 0],  # +Y
            [0, 0, axis_length],  # +Z
        ])

        R = np.array(tag_pose['rotation'])
        t = np.array(tag_pose['translation']).reshape(1, 3)

        axes_cam = (R @ axes_3d.T).T + t


        pts = (cam_mtx @ axes_cam.T).T
        pts = pts[:, :2] / pts[:, 2:3]

        axis_2d_pts = pts.astype(int)

        return axis_2d_pts










